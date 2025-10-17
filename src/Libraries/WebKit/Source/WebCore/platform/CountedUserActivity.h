/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#ifndef CountedUserActivity_h
#define CountedUserActivity_h

#include "UserActivity.h"

namespace WebCore {

class CountedUserActivity {
public:
    explicit CountedUserActivity(ASCIILiteral description)
        : m_activity(description)
    {
    }

    void increment()
    {
        if (!m_count++)
            m_activity.start();
    }

    void decrement()
    {
        if (!--m_count)
            m_activity.stop();
    }

private:
    UserActivity m_activity;
    size_t m_count { 0 };
};

} // namespace WebCore

using WebCore::CountedUserActivity;

#endif // CountedUserActivity_h
