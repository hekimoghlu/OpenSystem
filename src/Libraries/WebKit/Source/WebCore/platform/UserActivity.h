/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#ifndef UserActivity_h
#define UserActivity_h

#include <pal/HysteresisActivity.h>

#if HAVE(NS_ACTIVITY)
#include <objc/objc.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>
OBJC_CLASS NSString;
#endif

namespace WebCore {

// The UserActivity type is used to indicate to the operating system that
// a user initiated or visible action is taking place, and as such that
// resources should be allocated to the process accordingly.
class UserActivity : public PAL::HysteresisActivity {
public:
    class Impl {
    public:
        explicit Impl(ASCIILiteral description);

        void beginActivity();
        void endActivity();

#if HAVE(NS_ACTIVITY)
    private:
        RetainPtr<id> m_activity;
        RetainPtr<NSString> m_description;
#endif
    };

    WEBCORE_EXPORT explicit UserActivity(ASCIILiteral);

private:
    void hysteresisUpdated(PAL::HysteresisState);

    Impl m_impl;
};

} // namespace WebCore

using WebCore::UserActivity;

#endif // UserActivity_h
