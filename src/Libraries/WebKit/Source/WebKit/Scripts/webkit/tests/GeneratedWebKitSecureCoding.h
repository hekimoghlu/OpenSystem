/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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
#include "CoreIPCTypes.h"
#include <wtf/cocoa/VectorCocoa.h>

#if USE(AVFOUNDATION)
OBJC_CLASS AVOutputContext;
#endif
OBJC_CLASS NSSomeFoundationType;
OBJC_CLASS class NSSomeOtherFoundationType;
#if ENABLE(DATA_DETECTION)
OBJC_CLASS DDScannerResult;
#endif

namespace WebKit {

#if USE(AVFOUNDATION)
class CoreIPCAVOutputContext {
public:
    CoreIPCAVOutputContext(AVOutputContext *);
    CoreIPCAVOutputContext(const RetainPtr<AVOutputContext>& object)
        : CoreIPCAVOutputContext(object.get()) { }

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCAVOutputContext, void>;

    CoreIPCAVOutputContext(
        RetainPtr<NSString>&&,
        RetainPtr<NSString>&&
    );

    RetainPtr<NSString> m_AVOutputContextSerializationKeyContextID;
    RetainPtr<NSString> m_AVOutputContextSerializationKeyContextType;
};
#endif

class CoreIPCNSSomeFoundationType {
public:
    CoreIPCNSSomeFoundationType(NSSomeFoundationType *);
    CoreIPCNSSomeFoundationType(const RetainPtr<NSSomeFoundationType>& object)
        : CoreIPCNSSomeFoundationType(object.get()) { }

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCNSSomeFoundationType, void>;

    CoreIPCNSSomeFoundationType(
        RetainPtr<NSString>&&,
        RetainPtr<NSNumber>&&,
        RetainPtr<NSNumber>&&,
        RetainPtr<NSArray>&&,
        RetainPtr<NSArray>&&,
        RetainPtr<NSDictionary>&&,
        RetainPtr<NSDictionary>&&
    );

    RetainPtr<NSString> m_StringKey;
    RetainPtr<NSNumber> m_NumberKey;
    RetainPtr<NSNumber> m_OptionalNumberKey;
    RetainPtr<NSArray> m_ArrayKey;
    RetainPtr<NSArray> m_OptionalArrayKey;
    RetainPtr<NSDictionary> m_DictionaryKey;
    RetainPtr<NSDictionary> m_OptionalDictionaryKey;
};

class CoreIPCclass NSSomeOtherFoundationType {
public:
    CoreIPCclass NSSomeOtherFoundationType(class NSSomeOtherFoundationType *);
    CoreIPCclass NSSomeOtherFoundationType(const RetainPtr<class NSSomeOtherFoundationType>& object)
        : CoreIPCclass NSSomeOtherFoundationType(object.get()) { }

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCclass NSSomeOtherFoundationType, void>;

    CoreIPCclass NSSomeOtherFoundationType(
        RetainPtr<NSDictionary>&&
    );

    RetainPtr<NSDictionary> m_DictionaryKey;
};

#if ENABLE(DATA_DETECTION)
class CoreIPCDDScannerResult {
public:
    CoreIPCDDScannerResult(DDScannerResult *);
    CoreIPCDDScannerResult(const RetainPtr<DDScannerResult>& object)
        : CoreIPCDDScannerResult(object.get()) { }

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCDDScannerResult, void>;

    CoreIPCDDScannerResult(
        RetainPtr<NSString>&&,
        RetainPtr<NSNumber>&&,
        RetainPtr<NSNumber>&&,
        Vector<RetainPtr<DDScannerResult>>&&,
        std::optional<Vector<RetainPtr<DDScannerResult>>>&&,
        Vector<std::pair<String, RetainPtr<Number>>>&&,
        std::optional<Vector<std::pair<String, RetainPtr<DDScannerResult>>>>&&,
        Vector<RetainPtr<NSData>>&&,
        Vector<RetainPtr<SecTrustRef>>&&
    );

    RetainPtr<NSString> m_StringKey;
    RetainPtr<NSNumber> m_NumberKey;
    RetainPtr<NSNumber> m_OptionalNumberKey;
    Vector<RetainPtr<DDScannerResult>> m_ArrayKey;
    std::optional<Vector<RetainPtr<DDScannerResult>>> m_OptionalArrayKey;
    Vector<std::pair<String, RetainPtr<Number>>> m_DictionaryKey;
    std::optional<Vector<std::pair<String, RetainPtr<DDScannerResult>>>> m_OptionalDictionaryKey;
    Vector<RetainPtr<NSData>> m_DataArrayKey;
    Vector<RetainPtr<SecTrustRef>> m_SecTrustArrayKey;
};
#endif

} // namespace WebKit

#endif // PLATFORM(COCOA)
