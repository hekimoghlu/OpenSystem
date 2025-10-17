/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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
#include <wtf/TZoneMalloc.h>

#ifdef __OBJC__
@interface NSObject (WebKitSecureCoding)
- (NSDictionary *)_webKitPropertyListData;
- (id)_initWithWebKitPropertyListData:(NSDictionary *)plist;
@end
#endif

namespace WebKit {

struct AuxiliaryProcessCreationParameters;

namespace SecureCoding {

const HashSet<String>* classNamesExemptFromSecureCodingCrash();
void applyProcessCreationParameters(const AuxiliaryProcessCreationParameters&);

} // namespace SecureCoding

#ifdef __OBJC__

#if !HAVE(WK_SECURE_CODING_NSURLREQUEST)
class CoreIPCSecureCoding {
WTF_MAKE_TZONE_ALLOCATED(CoreIPCSecureCoding);
public:
    CoreIPCSecureCoding(id);
    CoreIPCSecureCoding(const RetainPtr<NSObject<NSSecureCoding>>& object)
        : CoreIPCSecureCoding(object.get())
    {
    }

    RetainPtr<id> toID() const { return m_secureCoding; }

    Class objectClass() { return m_secureCoding.get().class; }

private:
    friend struct IPC::ArgumentCoder<CoreIPCSecureCoding, void>;

    IPC::CoreIPCRetainPtr<NSObject<NSSecureCoding>> m_secureCoding;
};
#endif // !HAVE(WK_SECURE_CODING_NSURLREQUEST)

bool conformsToWebKitSecureCoding(id);

#endif // __OBJC__

} // namespace WebKit

#endif // PLATFORM(COCOA)
