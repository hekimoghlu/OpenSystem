/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifdef KERNEL
#define _KERNEL

#include <sys/buf.h>
#include <sys/kpi_mbuf.h>
#include <sys/mount.h>
#include <sys/namei.h>
#include <sys/ubc.h>
#include <miscfs/specfs/specdev.h>
#include <miscfs/devfs/devfs.h>
#include <kern/thread.h>
#include <kern/thread_call.h>
#include <kern/kalloc.h>
#include <string.h>


#define M_SMBFSHASH M_TEMP /* HACK XXX CSM, this type is used by the hashinit and hashdestroy functions */

#define SMB_MALLOC_TYPE(addr, type, flags) do { addr = kalloc_type(type, flags); } while(0)
#define SMB_MALLOC_TYPE_COUNT(addr, type, count, flags) do { addr = kalloc_type(type, count, flags); } while(0)
#define SMB_MALLOC_DATA(addr, size, flags) do { addr = kalloc_data(size, flags); } while(0)

#define SMB_FREE_TYPE(type, addr) do { kfree_type(type,addr); } while(0)
#define SMB_FREE_TYPE_COUNT(type, count, addr) do { kfree_type(type, count, addr); } while(0)
#define SMB_FREE_DATA(addr, size) do { kfree_data(addr,size); } while(0)

#undef FB_CURRENT

/* Max number of times we will attempt to open in a reconnect */
#define SMB_MAX_REOPEN_CNT  25

typedef enum modeventtype {
    MOD_LOAD,
    MOD_UNLOAD,
    MOD_SHUTDOWN
} modeventtype_t;

typedef struct kmod_info *module_t;

typedef int (*modeventhand_t)(module_t mod, int what, void *arg);

typedef struct moduledata {
    const char      *name;  /* module name */
    modeventhand_t  evhand; /* event handler */
    void            *priv;  /* extra data */
} moduledata_t;

#define DECLARE_MODULE(name, data, sub, order)      \
    moduledata_t * _smb_md_##name = &data;
#define SEND_EVENT(name, event)                     \
    {                                               \
        extern moduledata_t * _smb_md_##name;       \
        if (_smb_md_##name)                         \
        _smb_md_##name->evhand(smbfs_kmod_infop,    \
                     event,                         \
                     _smb_md_##name->priv);         \
    }
#define DEV_MODULE(name, evh, arg)                  \
    static moduledata_t name##_mod = {              \
        #name,                                      \
        evh,                                        \
        arg                                         \
    };                                              \
    DECLARE_MODULE(name, name##_mod, SI_SUB_DRIVERS, SI_ORDER_ANY);

struct smbnode;
extern int smb_smb_flush __P((struct smbnode *, vfs_context_t));

typedef int vnop_t __P((void *));

#define vn_todev(vp) (vnode_vtype(vp) == VBLK || vnode_vtype(vp) == VCHR ? \
              vnode_specrdev(vp) : NODEV)

void timevaladd(struct timeval *, struct timeval *);
void timevalsub(struct timeval *, struct timeval *);
#define timevalcmp(l, r, cmp)    timercmp(l, r, cmp)
#define timespeccmp(tvp, uvp, cmp)                  \
    (((tvp)->tv_sec == (uvp)->tv_sec) ?             \
        ((tvp)->tv_nsec cmp (uvp)->tv_nsec) :       \
        ((tvp)->tv_sec cmp (uvp)->tv_sec))
#define timespecadd(vvp, uvp)                       \
    do {                                            \
        (vvp)->tv_sec += (uvp)->tv_sec;             \
        (vvp)->tv_nsec += (uvp)->tv_nsec;           \
        if ((vvp)->tv_nsec >= 1000000000) {         \
            (vvp)->tv_sec++;                        \
            (vvp)->tv_nsec -= 1000000000;           \
        }                                           \
    } while (0)
#define timespecsub(vvp, uvp)                       \
    do {                                            \
        (vvp)->tv_sec -= (uvp)->tv_sec;             \
        (vvp)->tv_nsec -= (uvp)->tv_nsec;           \
        if ((vvp)->tv_nsec < 0) {                   \
            (vvp)->tv_sec--;                        \
            (vvp)->tv_nsec += 1000000000;           \
        }                                           \
    } while (0)

#endif /* KERNEL */
