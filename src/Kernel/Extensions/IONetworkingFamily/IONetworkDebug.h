/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#ifndef _IONETWORKDEBUG_H
#define _IONETWORKDEBUG_H

extern uint32_t gIONetworkDebugFlags;

enum {
    kIONF_kprintf   = 0x01,
    kIONF_IOLog     = 0x02
};

#if DEVELOPMENT
#define DLOG(fmt, args...)                              \
        do {                                            \
            if (gIONetworkDebugFlags & kIONF_kprintf)   \
                kprintf(fmt, ## args);                  \
            if (gIONetworkDebugFlags & kIONF_IOLog)     \
                IOLog(fmt, ## args);                    \
        } while (0)
#else
#define DLOG(fmt, args...)
#endif

#define LOG(fmt, args...)  \
        do { kprintf(fmt, ## args); IOLog(fmt, ## args); } while(0)

#endif /* _IONETWORKDEBUG_H */
