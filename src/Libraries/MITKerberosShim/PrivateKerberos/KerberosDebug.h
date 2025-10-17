/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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
#ifndef KERBEROSDEBUG_H
#define KERBEROSDEBUG_H

#include <stdarg.h>
#include <sys/types.h>
#include <mach/mach.h>

#ifdef __cplusplus
extern "C" {
#endif
    
/* 
 * These symbols will be exported for use by Kerberos tools.
 * Give them names that won't collide with other applications
 * linking against the Kerberos framework.
 */

#define SetSignalAction_(inAction)
#ifdef __PowerPlant__
#define GetSignalAction_() debugAction_Nothing
#else
#define GetSignalAction_() (0)
#endif

#ifdef __PowerPlant__
#	undef SignalPStr_
#	undef SignalCStr_
#	undef SignalIf_
#	undef SignalIfNot_
#endif /* __PowerPlant */

#define SignalPStr_(pstr)                                            \
    do {                                                             \
        dprintf ("%.*s in %s() (%s:%d)",                             \
                 (pstr) [0], (pstr) + 1,                             \
                 __FUNCTION__, __FILE__, __LINE__);                  \
    } while (0)

#define SignalCStr_(cstr)                                            \
    do {                                                             \
        dprintf ("%s in %s() (%s:%d)",                               \
                 cstr, __FUNCTION__, __FILE__, __LINE__);            \
    } while (0)

#define SignalIf_(test)                                              \
    do {                                                             \
        if (test) SignalCStr_("Assertion " #test " failed");         \
    } while (0)

#define SignalIfNot_(test) SignalIf_(!(test))

#define Assert_(test)      SignalIfNot_(test)

enum { errUncaughtException = 666 };

#define SafeTry_               try
#define SafeCatch_             catch (...)
#define SafeCatchOSErr_(error) catch (...) { SignalCStr_ ("Uncaught exception"); error = errUncaughtException; }

#define DebugThrow_(e)                                               \
    do {                                                             \
        dprintf ("Exception thrown from %s() (%s:%d)",               \
                 __FUNCTION__, __FILE__, __LINE__);                  \
        throw (e);                                                   \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif /* KERBEROSDEBUG_H */
