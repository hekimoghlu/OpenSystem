/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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

#include <wtf/ArgumentCoder.h>
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class CoreIPCDateComponents {
public:
    CoreIPCDateComponents(NSDateComponents *);
    RetainPtr<id> toID() const;

    static bool hasCorrectNumberOfComponentValues(const Vector<NSInteger>&);

private:
    friend struct IPC::ArgumentCoder<CoreIPCDateComponents, void>;
    CoreIPCDateComponents()
    {
    };

    String m_calendarIdentifier;
    String m_timeZoneName;
    Vector<NSInteger> m_componentValues;
};

} // namespace WebKit
