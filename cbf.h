/*
 * cbf.h -- Shared-memory counting Bloom filter for Linux
 *
 * Probabilistic set membership WITH DELETE and occurrence counts: tells you
 * whether an item is "definitely not" or "probably" in the set, in a fixed
 * amount of memory, with a tunable false-positive rate and no false negatives.
 * Each item is hashed once (XXH3-128); the two 64-bit halves drive k probes
 * (Kirsch-Mitzenmacher double hashing) into a power-of-two array of 4-bit
 * saturating counters. add increments the k counters, remove decrements them,
 * contains is true when all k are nonzero, and count_of returns their minimum
 * (an occurrence estimate 0..15). The array lives in a shared mapping so several
 * processes share one filter; a write-preferring futex rwlock with reader-slot
 * dead-process recovery guards mutation. Two filters of equal geometry can be
 * merged (counter-wise saturating add -> union with summed counts).
 *
 * Layout: Header -> reader_slots[1024] -> counters[m_ctr/2]  (two 4-bit per byte)
 */

#ifndef CBF_H
#define CBF_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <pthread.h>

#define XXH_INLINE_ALL
#include "xxhash.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "cbf.h: requires little-endian architecture"
#endif


/* ================================================================
 * Constants
 * ================================================================ */

#define CBF_MAGIC        0x53464243U  /* "CBFS" (little-endian) */
#define CBF_VERSION      1
#define CBF_ERR_BUFLEN   256
#ifndef CBF_READER_SLOTS
#define CBF_READER_SLOTS 1024         /* max concurrent reader processes for dead-process recovery */
#endif
#define CBF_MIN_CTR     64           /* floor on the counter count (power of two) */
#define CBF_MAX_CTR     0x4000000000ULL /* 2^38 counters = 128 GiB counter-array cap (4-bit) */
#define CBF_CTR_MAX     15           /* 4-bit saturating counter ceiling */
#define CBF_MIN_K        1
#define CBF_MAX_K        32

#define CBF_ERR(fmt, ...) do { if (errbuf) snprintf(errbuf, CBF_ERR_BUFLEN, fmt, ##__VA_ARGS__); } while (0)

/* ================================================================
 * Structs
 * ================================================================ */

/* Per-process slot for dead-process recovery.  Each shared rwlock counter
 * (the main rwlock-reader count, rwlock_waiters, rwlock_writers_waiting)
 * is mirrored here so a wrlock timeout can attribute and reverse a dead
 * process's contribution instead of waiting for the slow per-op timeout
 * drain. */
typedef struct {
    uint32_t pid;            /* 0 = unclaimed */
    uint32_t subcount;       /* in-flight rdlock acquisitions for this process */
    uint32_t waiters_parked; /* contribution to hdr->rwlock_waiters         */
    uint32_t writers_parked; /* contribution to hdr->rwlock_writers_waiting */
} CbfReaderSlot;

struct CbfHeader {
    uint32_t magic, version;          /* 0,4 */
    uint32_t k;                       /* 8   number of hash probes per item */
    uint32_t _pad0;                   /* 12 */
    uint64_t m_ctr;                   /* 16  counter count m (power of two) */
    uint64_t m_mask;                  /* 24  m_ctr - 1 (probe index mask) */
    uint64_t capacity;                /* 32  configured item capacity n (for stats) */
    double   fp_rate;                 /* 40  configured target false-positive rate (for stats) */
    uint64_t total_size;              /* 48 */
    uint64_t reader_slots_off;        /* 56 */
    uint64_t ctr_off;                /* 64 */
    uint32_t rwlock;                  /* 72 */
    uint32_t rwlock_waiters;          /* 76 */
    uint32_t rwlock_writers_waiting;  /* 80 */
    uint32_t slotless_readers;  /* live readers holding the lock with no reader-slot (was padding) */
    uint64_t stat_ops;                /* 88 */
    uint8_t  _pad[160];               /* 96..255 */
};
typedef struct CbfHeader CbfHeader;

_Static_assert(sizeof(CbfHeader) == 256, "CbfHeader must be 256 bytes");

/* ---- Process-local handle ---- */

typedef struct CbfHandle {
    CbfHeader     *hdr;
    CbfReaderSlot *reader_slots;  /* CBF_READER_SLOTS entries */
    void         *base;          /* mmap base */
    uint64_t      ctr_off;      /* validated counter-array offset, cached: never re-read from the peer-writable header */
    size_t        mmap_size;
    char         *path;          /* backing file path (strdup'd) */
    int           backing_fd;    /* memfd or reopened-fd to close on destroy, -1 for file/anon */
    uint32_t      my_slot_idx;   /* UINT32_MAX if all slots taken (no recovery for this handle) */
    uint32_t      cached_pid;    /* getpid() cached at last slot claim */
    uint32_t      cached_fork_gen; /* cbf_fork_gen value at last slot claim */
    uint32_t slotless_held; /* rwlock read-locks held with no reader-slot */
} CbfHandle;

/* ================================================================
 * Futex-based write-preferring read-write lock
 * with reader-slot dead-process recovery
 * ================================================================ */

#define CBF_RWLOCK_SPIN_LIMIT 32
#define CBF_LOCK_TIMEOUT_SEC  2  /* FUTEX_WAIT timeout for stale lock detection */

static inline void cbf_rwlock_spin_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

/* Extract writer PID from rwlock value (lower 31 bits when write-locked). */
#define CBF_RWLOCK_WRITER_BIT 0x80000000U
#define CBF_RWLOCK_PID_MASK   0x7FFFFFFFU
#define CBF_RWLOCK_WR(pid)    (CBF_RWLOCK_WRITER_BIT | ((uint32_t)(pid) & CBF_RWLOCK_PID_MASK))

/* Check if a PID is alive. Returns 1 if alive or unknown, 0 if definitely dead. */
/* Liveness via kill(pid,0). NOTE: cannot detect PID reuse -- if a dead
 * lock-holder's PID is recycled to an unrelated live process before recovery
 * runs, this reports "alive" and that slot's orphaned contribution is not
 * reclaimed until the recycled process exits. Robust detection would require
 * a per-slot process-start-time epoch (a header-layout/version change).
 * Documented under "Crash Safety" in the POD. */
static inline int cbf_pid_alive(uint32_t pid) {
    if (pid == 0) return 1; /* no owner recorded, assume alive */
    return !(kill((pid_t)pid, 0) == -1 && errno == ESRCH);
}

/* Force-recover a stale write lock left by a dead process.
 * CAS to OUR pid to hold the lock while fixing shared state, then release.
 * Using our pid (not a bare WRITER_BIT sentinel) means a subsequent
 * recovering process can detect and re-recover if we crash mid-recovery. */
static inline void cbf_recover_stale_lock(CbfHandle *h, uint32_t observed_rwlock) {
    CbfHeader *hdr = h->hdr;
    uint32_t mypid = CBF_RWLOCK_WR((uint32_t)getpid());
    if (!__atomic_compare_exchange_n(&hdr->rwlock, &observed_rwlock,
            mypid, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return;
    /* We now hold the write lock as mypid.  No additional shared state needs
     * repair here (this module has no seqlock); just release the lock. */
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static const struct timespec cbf_lock_timeout = { CBF_LOCK_TIMEOUT_SEC, 0 };

/* Process-global fork-generation counter.  Incremented in the pthread_atfork
 * child callback so every open handle detects a fork transition on the next
 * lock call without paying a getpid() syscall on the hot path. */
static uint32_t cbf_fork_gen = 1;
static pthread_once_t cbf_atfork_once = PTHREAD_ONCE_INIT;
static void cbf_on_fork_child(void) {
    __atomic_add_fetch(&cbf_fork_gen, 1, __ATOMIC_RELAXED);
}
static void cbf_atfork_init(void) {
    pthread_atfork(NULL, NULL, cbf_on_fork_child);
}

/* Ensure this process owns a reader slot.  Called from the lock helpers so
 * that fork()'d children pick up their own slot lazily instead of sharing
 * the parent's.  Hot-path is a single relaxed load + compare; only on a
 * fork-generation mismatch do we touch getpid() and scan slots. */
static inline void cbf_claim_reader_slot(CbfHandle *h) {
    uint32_t cur_gen = __atomic_load_n(&cbf_fork_gen, __ATOMIC_RELAXED);
    if (__builtin_expect(cur_gen == h->cached_fork_gen && h->my_slot_idx != UINT32_MAX, 1))
        return;
    /* Cold path -- register the atfork hook once per process, then claim. */
    pthread_once(&cbf_atfork_once, cbf_atfork_init);
    /* Re-read after pthread_once: cbf_on_fork_child may have bumped it. */
    cur_gen = __atomic_load_n(&cbf_fork_gen, __ATOMIC_RELAXED);
    uint32_t now_pid = (uint32_t)getpid();
    h->cached_pid = now_pid;
    if (cur_gen != h->cached_fork_gen) h->slotless_held = 0;  /* fork: child holds none of the parent's slotless read locks */
    h->cached_fork_gen = cur_gen;
    h->my_slot_idx = UINT32_MAX;
    uint32_t start = now_pid % CBF_READER_SLOTS;
    for (uint32_t i = 0; i < CBF_READER_SLOTS; i++) {
        uint32_t s = (start + i) % CBF_READER_SLOTS;
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(&h->reader_slots[s].pid,
                &expected, now_pid, 0,
                __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            /* Zero all mirror fields, not just subcount: a SIGKILL'd
             * predecessor may have left waiters_parked/writers_parked
             * non-zero, and cbf_recover_dead_readers won't drain them
             * once we own the slot (the CAS expects the dead PID). */
            __atomic_store_n(&h->reader_slots[s].subcount, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].waiters_parked, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].writers_parked, 0, __ATOMIC_RELAXED);
            h->my_slot_idx = s;
            return;
        }
    }
    /* Table full -- leave my_slot_idx = UINT32_MAX so we silently skip
     * tracking for this handle (lock still works; just no recovery). */
}

/* Atomically subtract `sub` from a counter, capped at 0 (never underflows). */
static inline void cbf_atomic_sub_cap(uint32_t *p, uint32_t sub) {
    if (!sub) return;
    uint32_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    for (;;) {
        uint32_t want = (cur > sub) ? cur - sub : 0;
        if (__atomic_compare_exchange_n(p, &cur, want,
                1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

/* Try to claim a dead slot (CAS pid -> 0) and drain its parked-waiter
 * contributions back to the global counters.  A no-op if the slot was stolen
 * by another recoverer or had no waiter contribution to drain.
 *
 * Note: subcount/waiters_parked/writers_parked are NOT zeroed here.
 * Between our CAS and a follow-up store, a new process could claim the
 * slot and start populating these fields -- our stores would clobber its
 * state.  cbf_claim_reader_slot zeros all three on every claim, so
 * leaving stale values is harmless. */
static inline void cbf_drain_dead_slot(CbfHandle *h, uint32_t i, uint32_t pid) {
    CbfHeader *hdr = h->hdr;
    uint32_t expected = pid;
    /* ACQ_REL on success: RELEASE publishes pid=0 to other observers;
     * ACQUIRE syncs us with prior writes from the dead process to
     * waiters_parked/writers_parked.  On weakly-ordered archs (aarch64)
     * a plain RELAXED load before the CAS could miss those writes;
     * loading them after the CAS keeps them inside the acquire window. */
    if (!__atomic_compare_exchange_n(&h->reader_slots[i].pid, &expected, 0,
            0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    uint32_t wp    = __atomic_load_n(&h->reader_slots[i].waiters_parked, __ATOMIC_RELAXED);
    uint32_t writp = __atomic_load_n(&h->reader_slots[i].writers_parked, __ATOMIC_RELAXED);
    if (wp)    cbf_atomic_sub_cap(&hdr->rwlock_waiters, wp);
    if (writp) cbf_atomic_sub_cap(&hdr->rwlock_writers_waiting, writp);
}

/* Scan reader slots for dead-process recovery.
 *
 * For each dead PID with non-zero contributions to the shared rwlock,
 * rwlock_waiters, or rwlock_writers_waiting counters, drain its share back
 * out so live processes don't have to wait for the slow per-op timeout
 * decrement to drain it for them.
 *
 * For the main rwlock counter we use the "no live reader holds -> force-
 * reset to 0" trick (precise) because per-process attribution of the
 * subcount is racy across the inc-counter-then-inc-subcount window. */
static inline void cbf_recover_dead_readers(CbfHandle *h) {
    if (!h->reader_slots) return;
    CbfHeader *hdr = h->hdr;
    int any_live_reader = 0;
    int found_dead_reader = 0;

    /* Pass 1: classify slots.  Slots with dead pid and sc == 0 (no rwlock
     * contribution to lose) are wiped immediately to free the slot for
     * future claimants and drain any orphan parked-waiter counters.  Slots
     * with dead pid and sc > 0 are left intact in this pass: if force-
     * reset cannot fire (because a live reader is concurrently present),
     * wiping the dead slot would lose the only record of its orphan
     * rwlock contribution and strand writers permanently once the live
     * reader releases. */
    for (uint32_t i = 0; i < CBF_READER_SLOTS; i++) {
        uint32_t pid = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
        if (pid == 0) continue;
        uint32_t sc = __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED);
        if (cbf_pid_alive(pid)) {
            if (sc > 0) any_live_reader = 1;
            continue;
        }
        if (sc > 0) { found_dead_reader = 1; continue; }
        cbf_drain_dead_slot(h, i, pid);
    }

    /* Pass 2: only if force-reset will fire.  Issue the rwlock force-
     * reset CAS FIRST, while the window since pass 1's last scan is
     * still narrow (a handful of instructions, as in the original
     * single-pass code).  A new reader that started rdlock between
     * pass 1's scan and the CAS will either:
     *   (a) have already CAS'd rwlock from cur to cur+1 -- our CAS then
     *       fails (cur mismatched), recovery yields and a future
     *       cycle retries; or
     *   (b) be still in the subcount-bump phase -- our CAS sees the
     *       stale cur and resets to 0; the new reader's subsequent CAS
     *       rwlock(0 -> 1) succeeds cleanly.
     * Only after the CAS resolves do we wipe the deferred dead slots,
     * keeping that work outside the race-sensitive window. */
    /* A live reader with no slot (table was full) is invisible to the scan
     * above but still holds a +1 in the lock word; never force-reset under it. */
    if (__atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0)
        any_live_reader = 1;
    if (found_dead_reader && !any_live_reader) {
        /* ACQUIRE: a late reader's subcount++ (before its rwlock CAS) is then visible below. */
        uint32_t cur = __atomic_load_n(&hdr->rwlock, __ATOMIC_ACQUIRE);
        int drain_ok = 1;   /* keep dead slots if the reset doesn't fire */
        if (cur > 0 && cur < CBF_RWLOCK_WRITER_BIT) {
            /* Re-scan for a live reader (fail-safe: only suppresses a reset). */
            int live_now = __atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0;
            for (uint32_t i = 0; !live_now && i < CBF_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p && cbf_pid_alive(p) &&
                    __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED) > 0)
                    live_now = 1;
            }
            if (live_now) {
                drain_ok = 0;
            } else if (__atomic_compare_exchange_n(&hdr->rwlock, &cur, 0,
                    0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
                if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
                    syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
            } else {
                drain_ok = 0;   /* rwlock changed under us -- shares may still be live */
            }
        }
        if (drain_ok) {
            for (uint32_t i = 0; i < CBF_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p == 0 || cbf_pid_alive(p)) continue;
                cbf_drain_dead_slot(h, i, p);
            }
        }
    }
}

/* Inspect the lock word after a futex-wait timeout.  If a dead writer
 * holds it, force-recover the lock.  Otherwise drain dead readers' shares
 * of the rwlock/waiter counters.  Called from rdlock and wrlock ETIMEDOUT
 * branches -- identical recovery logic in both. */
static inline void cbf_recover_after_timeout(CbfHandle *h) {
    CbfHeader *hdr = h->hdr;
    uint32_t val = __atomic_load_n(&hdr->rwlock, __ATOMIC_RELAXED);
    if (val >= CBF_RWLOCK_WRITER_BIT) {
        uint32_t pid = val & CBF_RWLOCK_PID_MASK;
        if (!cbf_pid_alive(pid))
            cbf_recover_stale_lock(h, val);
    } else {
        cbf_recover_dead_readers(h);
    }
}

/* Park/unpark helpers: bump the global waiter counters together with this
 * process's mirrored slot counters so a wrlock-timeout recovery scan can
 * attribute and reverse a dead PID's contribution.  Kept paired to make
 * accidental drift between global and per-slot counts impossible. */
static inline void cbf_park_reader(CbfHandle *h) {
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
}
static inline void cbf_unpark_reader(CbfHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
}
static inline void cbf_park_writer(CbfHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
}
static inline void cbf_unpark_writer(CbfHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
}

/* Reader accounting: a reader mirrors its +1 in the lock word so dead-reader
 * recovery can see it. A slotted reader uses its slot subcount; a reader that
 * could not claim a slot (table full) uses the global hdr->slotless_readers,
 * so recovery's force-reset never fires out from under it. leave() peels
 * slotless first so a later slot claim cannot misattribute the decrement. */
static inline void cbf_reader_enter(CbfHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_add_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
        h->slotless_held++;
    }
}
static inline void cbf_reader_leave(CbfHandle *h) {
    if (h->slotless_held > 0) {
        h->slotless_held--;
        __atomic_sub_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
    } else if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    }
}

static inline void cbf_rwlock_rdlock(CbfHandle *h) {
    cbf_claim_reader_slot(h);
    CbfHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    uint32_t *writers_waiting = &hdr->rwlock_writers_waiting;
    /* Claim subcount BEFORE bumping the shared rwlock counter.  This way
     * a concurrent writer-side recovery scan that sees our PID alive with
     * subcount > 0 will (correctly) defer force-reset, even while we are
     * still spinning trying to win the rwlock CAS.  Without this, a reader
     * killed between rwlock CAS-success and subcount++ would let recovery
     * force-reset rwlock to 0 underneath us, causing a UINT32_MAX wrap on
     * our eventual rdunlock dec. */
    cbf_reader_enter(h);
    for (int spin = 0; ; spin++) {
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Write-preferring: when lock is free (cur==0) and writers are
         * waiting, yield to let the writer acquire. When readers are
         * already active (cur>=1), new readers may join freely. */
        if (cur > 0 && cur < CBF_RWLOCK_WRITER_BIT) {
            if (__atomic_compare_exchange_n(lock, &cur, cur + 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        } else if (cur == 0 && !__atomic_load_n(writers_waiting, __ATOMIC_RELAXED)) {
            if (__atomic_compare_exchange_n(lock, &cur, 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        }
        if (__builtin_expect(spin < CBF_RWLOCK_SPIN_LIMIT, 1)) {
            cbf_rwlock_spin_pause();
            continue;
        }
        cbf_park_reader(h);
        cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Sleep when write-locked OR when yielding to waiting writers */
        if (cur >= CBF_RWLOCK_WRITER_BIT || cur == 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &cbf_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                cbf_unpark_reader(h);
                cbf_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        cbf_unpark_reader(h);
        spin = 0;
    }
}

static inline void cbf_rwlock_rdunlock(CbfHandle *h) {
    CbfHeader *hdr = h->hdr;
    /* Release the shared counter BEFORE dropping our subcount so that
     * "any live PID with subcount > 0" is a reliable in-flight indicator
     * for the writer-side recovery scan.  Inverting these would create a
     * window where we still own a unit of rwlock but our slot subcount is
     * 0, letting recovery force-reset rwlock underneath us. */
    uint32_t after = __atomic_sub_fetch(&hdr->rwlock, 1, __ATOMIC_RELEASE);
    cbf_reader_leave(h);
    if (after == 0 && __atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static inline void cbf_rwlock_wrlock(CbfHandle *h) {
    cbf_claim_reader_slot(h);  /* refresh cached_pid across fork */
    CbfHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    /* Encode PID in the rwlock word itself (0x80000000 | pid) to eliminate
     * any crash window between acquiring the lock and storing the owner. */
    uint32_t mypid = CBF_RWLOCK_WR(h->cached_pid);
    for (int spin = 0; ; spin++) {
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(lock, &expected, mypid,
                1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            return;
        if (__builtin_expect(spin < CBF_RWLOCK_SPIN_LIMIT, 1)) {
            cbf_rwlock_spin_pause();
            continue;
        }
        cbf_park_writer(h);
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        if (cur != 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &cbf_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                cbf_unpark_writer(h);
                cbf_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        cbf_unpark_writer(h);
        spin = 0;
    }
}

static inline void cbf_rwlock_wrunlock(CbfHandle *h) {
    CbfHeader *hdr = h->hdr;
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

/* ================================================================
 * Layout math + create / open / destroy
 *
 * Layout: Header -> reader_slots[1024] -> counters[m_ctr/2]  (two 4-bit per byte)
 * ================================================================ */

/* Single source of truth for the mmap region layout offsets. */
typedef struct { uint64_t reader_slots, counters; } CbfLayout;

static inline CbfLayout cbf_layout(void) {
    CbfLayout L;
    L.reader_slots = sizeof(CbfHeader);
    L.counters     = L.reader_slots + (uint64_t)CBF_READER_SLOTS * sizeof(CbfReaderSlot);
    L.counters     = (L.counters + 7) & ~(uint64_t)7;   /* 8-byte align the counter array */
    return L;
}

static inline uint64_t cbf_total_size(uint64_t m_ctr) {
    CbfLayout L = cbf_layout();
    return L.counters + (m_ctr / 2);   /* 4-bit counters, two per byte; m_ctr is a power of two >= 64 -> exact bytes */
}

/* round v up to the next power of two (64-bit), with a floor of CBF_MIN_CTR */
static inline uint64_t cbf_next_pow2_u64(uint64_t v) {
    if (v <= CBF_MIN_CTR) return CBF_MIN_CTR;
    return 1ULL << (64 - __builtin_clzll(v - 1));
}

static inline void cbf_init_header(void *base, uint32_t k, uint64_t m_ctr,
                                  uint64_t capacity, double fp_rate, uint64_t total) {
    CbfLayout L = cbf_layout();
    CbfHeader *hdr = (CbfHeader *)base;
    /* Explicitly zero the header + reader-slot region (lock-recovery state); the
       counter array relies on the fresh mapping being OS zero-filled. */
    memset(base, 0, (size_t)L.counters);
    hdr->magic            = CBF_MAGIC;
    hdr->version          = CBF_VERSION;
    hdr->k                = k;
    hdr->m_ctr            = m_ctr;
    hdr->m_mask           = m_ctr - 1;
    hdr->capacity         = capacity;
    hdr->fp_rate          = fp_rate;
    hdr->total_size       = total;
    hdr->reader_slots_off = L.reader_slots;
    hdr->ctr_off          = L.counters;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

/* ---- 4-bit saturating counters: two per byte, low nibble = even index ---- */
static inline uint8_t *cbf_counters(CbfHandle *h) {
    return (uint8_t *)((char *)h->base + h->ctr_off);
}
static inline uint32_t cbf_ctr_get(const uint8_t *c, uint64_t i) {
    return (c[i >> 1] >> ((unsigned)(i & 1) << 2)) & 0xFu;
}
static inline void cbf_ctr_set(uint8_t *c, uint64_t i, uint32_t v) {
    unsigned sh = (unsigned)(i & 1) << 2;
    c[i >> 1] = (uint8_t)((c[i >> 1] & ~(0xFu << sh)) | ((v & 0xFu) << sh));
}

/* Layer B trusted bound: the number of counter BYTES guaranteed to lie within
 * the real mapping.  Derived from the process-local mmap_size (fixed at attach
 * time, not peer-writable) and the SAME ctr_off cbf_counters() uses, so a peer
 * sharing the backing file that corrupts the header (m_ctr / m_mask / ctr_off)
 * after attach-time validation can never drive an access outside the mapping.
 * For a valid filter this equals m_ctr/2 exactly, so every clamp against it
 * below is a never-taken branch in normal use. */
static inline uint64_t cbf_ctr_bytes_max(CbfHandle *h) {
    uint64_t off = h->ctr_off;
    if (off >= h->mmap_size) return 0;
    return h->mmap_size - off;
}

static inline CbfHandle *cbf_setup(void *base, size_t map_size,
                                 const char *path, int backing_fd) {
    CbfHeader *hdr = (CbfHeader *)base;
    CbfHandle *h = (CbfHandle *)calloc(1, sizeof(CbfHandle));
    if (!h) {
        munmap(base, map_size);
        if (backing_fd >= 0) close(backing_fd);
        return NULL;
    }
    h->hdr          = hdr;
    h->base         = base;
    h->reader_slots = (CbfReaderSlot *)((uint8_t *)base + hdr->reader_slots_off);
    h->ctr_off     = hdr->ctr_off;   /* single validated read; bound and pointer stay consistent */
    h->mmap_size    = map_size;
    h->path         = path ? strdup(path) : NULL;
    h->backing_fd   = backing_fd;
    h->my_slot_idx  = UINT32_MAX;
    return h;
}

/* Validate a mapped header (shared by cbf_create reopen and cbf_open_fd). */
static inline int cbf_validate_header(const CbfHeader *hdr, uint64_t file_size) {
    if (hdr->magic != CBF_MAGIC) return 0;
    if (hdr->version != CBF_VERSION) return 0;
    if (hdr->k < CBF_MIN_K || hdr->k > CBF_MAX_K) return 0;
    if (hdr->m_ctr < CBF_MIN_CTR || hdr->m_ctr > CBF_MAX_CTR) return 0;
    if ((hdr->m_ctr & (hdr->m_ctr - 1)) != 0) return 0;        /* power of two */
    if (hdr->m_mask != hdr->m_ctr - 1) return 0;
    if (hdr->capacity == 0) return 0;
    if (!(hdr->fp_rate > 0.0 && hdr->fp_rate < 1.0)) return 0;
    if (hdr->total_size != file_size) return 0;
    if (hdr->total_size != cbf_total_size(hdr->m_ctr)) return 0;
    CbfLayout L = cbf_layout();
    if (hdr->reader_slots_off != L.reader_slots) return 0;
    if (hdr->ctr_off != L.counters) return 0;
    return 1;
}

/* validate args + compute the geometry (k, m_ctr) */
static int cbf_validate_create_args(uint64_t capacity, double fp_rate,
                                   uint32_t *k_out, uint64_t *m_bits_out, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    if (capacity < 1) { CBF_ERR("capacity must be >= 1"); return 0; }
    if (!(fp_rate > 0.0 && fp_rate < 1.0)) { CBF_ERR("fp_rate must be between 0 and 1 (exclusive)"); return 0; }

    /* k = round(-log2(fp_rate)) clamped to [1, 32] */
    long kl = lround(-log2(fp_rate));
    if (kl < CBF_MIN_K) kl = CBF_MIN_K;
    if (kl > CBF_MAX_K) kl = CBF_MAX_K;
    uint32_t k = (uint32_t)kl;

    /* m_opt = ceil(capacity * k / ln2); reject if it would exceed the counter-
     * array cap (otherwise the filter would be silently undersized -> fp_rate
     * broken); m_ctr = next_pow2(m_opt), floor CBF_MIN_CTR. */
    double m_opt_d = ceil((double)capacity * (double)k / M_LN2);
    if (m_opt_d > (double)CBF_MAX_CTR) { CBF_ERR("capacity too large for the counter-array cap"); return 0; }
    uint64_t m_ctr = cbf_next_pow2_u64((uint64_t)m_opt_d);

    *k_out = k;
    *m_bits_out = m_ctr;
    return 1;
}

/* Securely obtain a fd for a path-backed segment: create it exclusively
 * (O_CREAT|O_EXCL|O_NOFOLLOW at `mode`, default 0600 = owner-only), or, if it
 * already exists, attach to it (O_RDWR|O_NOFOLLOW, no O_CREAT). O_EXCL blocks a
 * pre-seeded or hard-linked file and O_NOFOLLOW a symlink swap, so a local
 * attacker can no longer redirect or poison the backing store through the path.
 * Cross-user sharing is opt-in via a wider `mode` (e.g. 0660); the caller still
 * validates the file's contents via cbf_validate_header. */
static int cbf_secure_open(const char *path, mode_t mode, char *errbuf) {
    for (int attempt = 0; attempt < 100; attempt++) {
        int fd = open(path, O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, mode);
        if (fd >= 0) { (void)fchmod(fd, mode); return fd; }   /* exact mode: umask narrowed the O_EXCL create */
        if (errno != EEXIST) { CBF_ERR("create %s: %s", path, strerror(errno)); return -1; }
        fd = open(path, O_RDWR|O_NOFOLLOW|O_CLOEXEC);
        if (fd >= 0) return fd;
        if (errno == ENOENT) continue;   /* creator unlinked between our two opens; retry */
        CBF_ERR("open %s: %s", path, strerror(errno));  /* ELOOP => symlink rejected */
        return -1;
    }
    CBF_ERR("open %s: create/attach kept racing", path);
    return -1;
}

static CbfHandle *cbf_create(const char *path, uint64_t capacity, double fp_rate, mode_t mode, char *errbuf) {
    uint32_t k;
    uint64_t m_ctr;
    if (!cbf_validate_create_args(capacity, fp_rate, &k, &m_ctr, errbuf)) return NULL;

    uint64_t total = cbf_total_size(m_ctr);
    int anonymous = (path == NULL);
    int fd = -1;
    size_t map_size;
    void *base;

    if (anonymous) {
        map_size = (size_t)total;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) { CBF_ERR("mmap: %s", strerror(errno)); return NULL; }
    } else {
        fd = cbf_secure_open(path, mode, errbuf);
        if (fd < 0) return NULL;
        if (flock(fd, LOCK_EX) < 0) { CBF_ERR("flock: %s", strerror(errno)); close(fd); return NULL; }
        struct stat st;
        if (fstat(fd, &st) < 0) { CBF_ERR("fstat: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        int is_new = (st.st_size == 0);
        if (!is_new && (uint64_t)st.st_size < sizeof(CbfHeader)) {
            CBF_ERR("%s: file too small (%lld)", path, (long long)st.st_size);
            flock(fd, LOCK_UN); close(fd); return NULL;
        }
        if (is_new && ftruncate(fd, (off_t)total) < 0) {
            CBF_ERR("ftruncate: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL;
        }
        map_size = is_new ? (size_t)total : (size_t)st.st_size;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { CBF_ERR("mmap: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        if (!is_new) {
            if (!cbf_validate_header((CbfHeader *)base, (uint64_t)st.st_size)) {
                CBF_ERR("invalid counting Bloom filter file"); munmap(base, map_size); flock(fd, LOCK_UN); close(fd); return NULL;
            }
            flock(fd, LOCK_UN); close(fd);
            return cbf_setup(base, map_size, path, -1);
        }
    }
    cbf_init_header(base, k, m_ctr, capacity, fp_rate, total);
    if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
    return cbf_setup(base, map_size, path, -1);
}

static CbfHandle *cbf_create_memfd(const char *name, uint64_t capacity, double fp_rate, char *errbuf) {
    uint32_t k;
    uint64_t m_ctr;
    if (!cbf_validate_create_args(capacity, fp_rate, &k, &m_ctr, errbuf)) return NULL;

    uint64_t total = cbf_total_size(m_ctr);
    int fd = memfd_create(name ? name : "bloom", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) { CBF_ERR("memfd_create: %s", strerror(errno)); return NULL; }
    if (ftruncate(fd, (off_t)total) < 0) {
        CBF_ERR("ftruncate: %s", strerror(errno)); close(fd); return NULL;
    }
    (void)fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
    void *base = mmap(NULL, (size_t)total, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { CBF_ERR("mmap: %s", strerror(errno)); close(fd); return NULL; }
    cbf_init_header(base, k, m_ctr, capacity, fp_rate, total);
    return cbf_setup(base, (size_t)total, NULL, fd);
}

static CbfHandle *cbf_open_fd(int fd, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    struct stat st;
    if (fstat(fd, &st) < 0) { CBF_ERR("fstat: %s", strerror(errno)); return NULL; }
    if ((uint64_t)st.st_size < sizeof(CbfHeader)) { CBF_ERR("too small"); return NULL; }
    size_t ms = (size_t)st.st_size;
    void *base = mmap(NULL, ms, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { CBF_ERR("mmap: %s", strerror(errno)); return NULL; }
    if (!cbf_validate_header((CbfHeader *)base, (uint64_t)st.st_size)) {
        CBF_ERR("invalid counting Bloom filter table"); munmap(base, ms); return NULL;
    }
    int myfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (myfd < 0) { CBF_ERR("fcntl: %s", strerror(errno)); munmap(base, ms); return NULL; }
    return cbf_setup(base, ms, NULL, myfd);
}

static void cbf_destroy(CbfHandle *h) {
    if (!h) return;
    /* Release our reader slot on clean teardown (else short-lived-reader churn
     * exhausts the slot table); skip if a lock is still held (subcount>0). */
    if (h->reader_slots && h->my_slot_idx != UINT32_MAX && h->cached_pid &&
        h->cached_fork_gen == __atomic_load_n(&cbf_fork_gen, __ATOMIC_RELAXED) &&
        __atomic_load_n(&h->reader_slots[h->my_slot_idx].subcount, __ATOMIC_ACQUIRE) == 0) {
        uint32_t expected = h->cached_pid;
        __atomic_compare_exchange_n(&h->reader_slots[h->my_slot_idx].pid,
                &expected, 0, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    if (h->backing_fd >= 0) close(h->backing_fd);
    if (h->base) munmap(h->base, h->mmap_size);
    free(h->path);
    free(h);
}

static inline int cbf_msync(CbfHandle *h) {
    if (!h || !h->base) return 0;
    return msync(h->base, h->mmap_size, MS_SYNC);
}

/* ================================================================
 * Counting Bloom filter operations (callers hold the lock) -- double hashing
 * (Kirsch-Mitzenmacher): one XXH3-128 hash drives all k probes, each of which
 * indexes a 4-bit saturating counter.  A probe idx reaches counter byte
 * (idx >> 1); the highest a masked probe can reach is (mask >> 1).
 * ================================================================ */

static inline void cbf_indices(CbfHandle *h, const void *item, size_t len,
                              uint64_t *h1, uint64_t *h2) {
    (void)h;
    XXH128_hash_t hh = XXH3_128bits(item, len);
    *h1 = hh.low64;
    *h2 = hh.high64 | 1ULL;   /* force odd so the k probes spread over the pow2 table */
}

/* increment k counters (saturating at 15); return 1 if the item was probably NEW
 * (at least one counter was 0 before), else 0.  If two of the k probes alias the
 * same counter it is incremented once per probe -- add/remove stay symmetric. */
static int cbf_add_locked(CbfHandle *h, const void *item, size_t len) {
    uint64_t h1, h2;
    cbf_indices(h, item, len, &h1, &h2);
    uint64_t mask = h->hdr->m_mask;
    uint32_t k = h->hdr->k;
    uint8_t *c = cbf_counters(h);
    /* Layer B: the highest counter BYTE any masked probe reaches is (mask >> 1).
     * m_mask lives in the shared segment, so a peer sharing the backing file
     * could widen it past the real counter array after validation; bound it
     * against the mapping.  For a valid filter (mask == m_ctr-1) this is
     * (m_ctr-1)/2 < m_ctr/2 bytes, so the branch is never taken. */
    if ((mask >> 1) >= cbf_ctr_bytes_max(h)) return 0;
    int was_new = 0;
    for (uint32_t i = 0; i < k; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) & mask;
        uint32_t v = cbf_ctr_get(c, idx);
        if (v == 0) was_new = 1;
        if (v < CBF_CTR_MAX) cbf_ctr_set(c, idx, v + 1);   /* saturate at 15 */
    }
    return was_new;
}

/* return 1 if ALL k counters are > 0 (probably present), else 0 */
static int cbf_contains_locked(CbfHandle *h, const void *item, size_t len) {
    uint64_t h1, h2;
    cbf_indices(h, item, len, &h1, &h2);
    uint64_t mask = h->hdr->m_mask;
    uint32_t k = h->hdr->k;
    const uint8_t *c = cbf_counters(h);
    /* Layer B: bound the highest reachable byte against the mapping (see
     * cbf_add_locked).  A corrupt mask cannot confirm membership -> "not
     * present" is the safe answer; never-taken for a valid filter. */
    if ((mask >> 1) >= cbf_ctr_bytes_max(h)) return 0;
    for (uint32_t i = 0; i < k; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) & mask;
        if (cbf_ctr_get(c, idx) == 0) return 0;
    }
    return 1;
}

/* estimated occurrence count of (item,len): the MIN of its k counters (0..15).
 * Collisions can only raise a counter, so min never under-counts a present item
 * (it is an upper estimate); saturates at CBF_CTR_MAX. (caller holds a lock) */
static int cbf_count_of_locked(CbfHandle *h, const void *item, size_t len) {
    uint64_t h1, h2;
    cbf_indices(h, item, len, &h1, &h2);
    uint64_t mask = h->hdr->m_mask;
    uint32_t k = h->hdr->k;
    const uint8_t *c = cbf_counters(h);
    if ((mask >> 1) >= cbf_ctr_bytes_max(h)) return 0;
    uint32_t mn = CBF_CTR_MAX;
    for (uint32_t i = 0; i < k; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) & mask;
        uint32_t v = cbf_ctr_get(c, idx);
        if (v < mn) mn = v;
        if (mn == 0) return 0;   /* absent -> short-circuit */
    }
    return (int)mn;
}

/* remove one occurrence of (item,len): if present (all k counters > 0),
 * decrement each (a saturated 15 stays stuck; never drops below 0) and return 1;
 * if not present, change nothing and return 0.
 * CAVEAT: like every counting Bloom filter, decrementing counters for an item
 * never added -- or one whose k probes collide with present items -- can push a
 * shared counter to 0 and cause a false negative for another item.  Only remove
 * items you actually added, and remove an item as many times as it was added. */
static int cbf_remove_locked(CbfHandle *h, const void *item, size_t len) {
    uint64_t h1, h2;
    cbf_indices(h, item, len, &h1, &h2);
    uint64_t mask = h->hdr->m_mask;
    uint32_t k = h->hdr->k;
    uint8_t *c = cbf_counters(h);
    if ((mask >> 1) >= cbf_ctr_bytes_max(h)) return 0;
    /* present-check FIRST: never decrement unless all k counters are > 0, so a
     * remove of an absent item is a true no-op (no false negatives introduced) */
    for (uint32_t i = 0; i < k; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) & mask;
        if (cbf_ctr_get(c, idx) == 0) return 0;
    }
    for (uint32_t i = 0; i < k; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) & mask;
        uint32_t v = cbf_ctr_get(c, idx);
        if (v > 0 && v < CBF_CTR_MAX) cbf_ctr_set(c, idx, v - 1);   /* saturated stays stuck */
    }
    return 1;
}

/* count nonzero counters across the whole array -- the counting analogue of a
 * Bloom popcount, used for the distinct-item estimate. (caller holds a lock) */
static uint64_t cbf_nonzero_count_locked(CbfHandle *h) {
    const uint8_t *c = cbf_counters(h);
    uint64_t bytes = h->hdr->m_ctr / 2;
    uint64_t bytes_max = cbf_ctr_bytes_max(h);   /* Layer B: clamp scan to the mapping */
    if (bytes > bytes_max) bytes = bytes_max;
    uint64_t n = 0;
    for (uint64_t b = 0; b < bytes; b++) {
        uint8_t byte = c[b];
        if (byte & 0x0F) n++;
        if (byte & 0xF0) n++;
    }
    return n;
}

/* estimate the number of distinct items added, from a pre-computed nonzero-counter
   count X.  n_est = -(m/k) * ln(1 - X/m); saturated -> capacity. (holds a lock) */
static uint64_t cbf_count_from_nonzero(CbfHandle *h, uint64_t X) {
    uint64_t m_ctr = h->hdr->m_ctr;
    uint32_t k = h->hdr->k;
    if (X >= m_ctr) return h->hdr->capacity;     /* saturated */
    double n_est = -((double)m_ctr / (double)k) * log(1.0 - (double)X / (double)m_ctr);
    if (n_est < 0.0) n_est = 0.0;
    return (uint64_t)(n_est + 0.5);
}

/* estimate the number of distinct items added (scans the array). (holds a lock) */
static uint64_t cbf_count_locked(CbfHandle *h) {
    return cbf_count_from_nonzero(h, cbf_nonzero_count_locked(h));
}

/* merge src counters into dst (caller guarantees equal m_ctr): counter-wise
 * saturating add.  src_bytes is how many counter bytes the src buffer holds. */
static void cbf_merge_counters(CbfHandle *dst, const uint8_t *src, uint64_t src_bytes) {
    uint8_t *c = cbf_counters(dst);
    uint64_t bytes = dst->hdr->m_ctr / 2;
    uint64_t bytes_max = cbf_ctr_bytes_max(dst);  /* Layer B: clamp writes to dst mapping */
    if (bytes > bytes_max) bytes = bytes_max;
    if (bytes > src_bytes) bytes = src_bytes;     /* ...and reads to the src buffer */
    for (uint64_t b = 0; b < bytes; b++) {
        uint32_t lo = (uint32_t)(c[b] & 0x0F) + (uint32_t)(src[b] & 0x0F);
        uint32_t hi = (uint32_t)((c[b] >> 4) & 0x0F) + (uint32_t)((src[b] >> 4) & 0x0F);
        if (lo > CBF_CTR_MAX) lo = CBF_CTR_MAX;   /* saturate each nibble at 15 */
        if (hi > CBF_CTR_MAX) hi = CBF_CTR_MAX;
        c[b] = (uint8_t)(lo | (hi << 4));
    }
}

/* reset all counters to 0 (caller holds the write lock) */
static inline void cbf_clear_locked(CbfHandle *h) {
    uint64_t bytes = h->hdr->m_ctr / 2;
    uint64_t bytes_max = cbf_ctr_bytes_max(h);    /* Layer B: clamp memset to the mapping */
    if (bytes > bytes_max) bytes = bytes_max;
    memset(cbf_counters(h), 0, (size_t)bytes);
}

#endif /* CBF_H */
