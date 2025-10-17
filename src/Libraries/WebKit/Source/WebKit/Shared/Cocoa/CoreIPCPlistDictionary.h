/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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

#include "CoreIPCPlistObject.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/KeyValuePair.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/Vector.h>

namespace WebKit {

class CoreIPCPlistDictionary {
    WTF_MAKE_TZONE_ALLOCATED(CoreIPCPlistDictionary);
public:
    CoreIPCPlistDictionary(NSDictionary *);
    CoreIPCPlistDictionary(const RetainPtr<NSDictionary>&);
    CoreIPCPlistDictionary(CoreIPCPlistDictionary&&);
    CoreIPCPlistDictionary& operator=(CoreIPCPlistDictionary&&) = default;
    ~CoreIPCPlistDictionary();

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCPlistDictionary, void>;

    using ValueType = Vector<KeyValuePair<CoreIPCString, CoreIPCPlistObject>>;

    CoreIPCPlistDictionary(ValueType&&);

    ValueType m_keyValuePairs;
};

} // namespace WebKit

#endif // PLATFORM(COCOA)
