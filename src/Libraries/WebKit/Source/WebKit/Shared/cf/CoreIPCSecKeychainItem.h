/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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

#if HAVE(SEC_KEYCHAIN)

#import <Security/SecKeychainItem.h>
#import <wtf/ProcessPrivilege.h>
#import <wtf/RetainPtr.h>
#import <wtf/cf/VectorCF.h>

namespace WebKit {

// For now, the only way to serialize/deserialize SecKeychainItem objects is via
// SecKeychainItemCreatePersistentReference()/SecKeychainItemCopyFromPersistentReference(). rdar://122050787

class CoreIPCSecKeychainItem {
public:
    CoreIPCSecKeychainItem(SecKeychainItemRef keychainItem)
        : m_persistentRef(persistentRefForKeychainItem(keychainItem))
    {
    }

    CoreIPCSecKeychainItem(RetainPtr<CFDataRef> data)
        : m_persistentRef(data)
    {
    }

    CoreIPCSecKeychainItem(std::span<const uint8_t> data)
        : m_persistentRef(data.empty() ? nullptr : adoptCF(CFDataCreate(kCFAllocatorDefault, data.data(), data.size())))
    {
    }

    RetainPtr<SecKeychainItemRef> createSecKeychainItem() const
    {
        RELEASE_ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessCredentials));

        if (!m_persistentRef)
            return nullptr;

        CFDataRef data = m_persistentRef.get();
        // SecKeychainItemCopyFromPersistentReference() cannot handle 0-length CFDataRefs.
        if (!CFDataGetLength(data))
            return nullptr;

        ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        SecKeychainItemRef keychainItem = NULL;
        SecKeychainItemCopyFromPersistentReference(data, &keychainItem);
        ALLOW_DEPRECATED_DECLARATIONS_END
        return adoptCF(keychainItem);
    }

    std::span<const uint8_t> dataReference() const
    {
        if (!m_persistentRef)
            return { };

        return span(m_persistentRef.get());
    }

private:
    RetainPtr<CFDataRef> persistentRefForKeychainItem(SecKeychainItemRef keychainItem) const
    {
        RELEASE_ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessCredentials));
        if (!keychainItem)
            return nullptr;

        ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        CFDataRef data = NULL;
        SecKeychainItemCreatePersistentReference(keychainItem, &data);
        ALLOW_DEPRECATED_DECLARATIONS_END

        return adoptCF(data);
    }

    RetainPtr<CFDataRef> m_persistentRef;
};

} // namespace WebKit

#endif // HAVE(SEC_KEYCHAIN)
