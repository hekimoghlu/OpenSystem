/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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

#include "ArgumentCodersCocoa.h"
#include <wtf/RetainPtr.h>

namespace WebKit {

class CoreIPCNSCFObject;

class CoreIPCArray {
public:
    CoreIPCArray(NSArray *);
    CoreIPCArray(const RetainPtr<NSArray>&);
    CoreIPCArray(CoreIPCArray&&);
    ~CoreIPCArray();

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCArray, void>;

    CoreIPCArray(Vector<CoreIPCNSCFObject>&&);

    Vector<CoreIPCNSCFObject> m_array;
};

} // namespace WebKit

#endif // PLATFORM(COCOA)
