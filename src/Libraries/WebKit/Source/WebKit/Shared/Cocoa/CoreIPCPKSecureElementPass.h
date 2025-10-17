/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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

#if USE(PASSKIT)

#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>

OBJC_CLASS PKSecureElementPass;

namespace WebKit {

class CoreIPCPKSecureElementPass {
public:
    CoreIPCPKSecureElementPass(PKSecureElementPass *);
    CoreIPCPKSecureElementPass(Vector<uint8_t>&& data)
        : m_data(WTFMove(data)) { }

    RetainPtr<id> toID() const;
    const Vector<uint8_t>& ipcData() const { return m_data; }

private:
    Vector<uint8_t> m_data;
};

} // namespace WebKit

#endif // USE(PASSKIT)
