/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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
#pragma once

#if PLATFORM(COCOA)

#import <CoreFoundation/CoreFoundation.h>
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebKit {

class CoreIPCDate {
public:

#ifdef __OBJC__
    CoreIPCDate(NSDate *date)
        : CoreIPCDate(bridge_cast(date))
    {
    }
#endif

    CoreIPCDate(CFDateRef date)
        : m_absoluteTime(CFDateGetAbsoluteTime(date))
    {
    }

    CoreIPCDate(const double absoluteTime)
        : m_absoluteTime(absoluteTime)
    {
    }

    RetainPtr<CFDateRef> createCFDate() const
    {
        return adoptCF(CFDateCreate(0, m_absoluteTime));
    }

    double get() const
    {
        return m_absoluteTime;
    }

    RetainPtr<id> toID() const
    {
        return bridge_cast(createCFDate().get());
    }

private:
    double m_absoluteTime;
};

}


#endif // PLATFORM(COCOA)
