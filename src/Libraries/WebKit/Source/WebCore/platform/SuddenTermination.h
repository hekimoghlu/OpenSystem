/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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
#ifndef SuddenTermination_h
#define SuddenTermination_h

#include <wtf/Noncopyable.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

    // Once disabled via one or more more calls to disableSuddenTermination(), fast shutdown
    // is not valid until enableSuddenTermination() has been called an equal number of times.
    // On Mac, these are thin wrappers around Mac OS X functions of the same name.
#if PLATFORM(MAC)
    WEBCORE_EXPORT void disableSuddenTermination();
    WEBCORE_EXPORT void enableSuddenTermination();
#else
    inline void disableSuddenTermination() { }
    inline void enableSuddenTermination() { }
#endif

    class SuddenTerminationDisabler {
        WTF_MAKE_TZONE_ALLOCATED_INLINE(SuddenTerminationDisabler);
        WTF_MAKE_NONCOPYABLE(SuddenTerminationDisabler);
    public:
        SuddenTerminationDisabler()
        {
            disableSuddenTermination();
        }

        ~SuddenTerminationDisabler()
        {
            enableSuddenTermination();
        }
    };

} // namespace WebCore

#endif // SuddenTermination_h
