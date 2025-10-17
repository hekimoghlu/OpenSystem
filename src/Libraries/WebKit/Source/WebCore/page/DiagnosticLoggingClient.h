/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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

#include "DiagnosticLoggingDomain.h"
#include <variant>
#include <wtf/CheckedPtr.h>
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum DiagnosticLoggingResultType : uint8_t;
enum class ShouldSample : bool { No, Yes };

struct DiagnosticLoggingDictionary {
    using Payload = std::variant<String, uint64_t, int64_t, bool, double>;
    using Dictionary = HashMap<String, Payload>;
    Dictionary dictionary;
    void set(String key, Payload value) { dictionary.set(WTFMove(key), WTFMove(value)); }
};

class DiagnosticLoggingClient : public CanMakeCheckedPtr<DiagnosticLoggingClient> {
    WTF_MAKE_TZONE_ALLOCATED(DiagnosticLoggingClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DiagnosticLoggingClient);
public:
    virtual void logDiagnosticMessage(const String& message, const String& description, ShouldSample) = 0;
    virtual void logDiagnosticMessageWithResult(const String& message, const String& description, DiagnosticLoggingResultType, ShouldSample) = 0;
    virtual void logDiagnosticMessageWithValue(const String& message, const String& description, double value, unsigned significantFigures, ShouldSample) = 0;
    virtual void logDiagnosticMessageWithEnhancedPrivacy(const String& message, const String& description, ShouldSample) = 0;

    using ValuePayload = DiagnosticLoggingDictionary::Payload;
    using ValueDictionary = DiagnosticLoggingDictionary;

    virtual void logDiagnosticMessageWithValueDictionary(const String& message, const String& description, const ValueDictionary&, ShouldSample) = 0;
    virtual void logDiagnosticMessageWithDomain(const String& message, DiagnosticLoggingDomain) = 0;

    static bool shouldLogAfterSampling(ShouldSample);

    virtual ~DiagnosticLoggingClient() = default;
};

inline bool DiagnosticLoggingClient::shouldLogAfterSampling(ShouldSample shouldSample)
{
    if (shouldSample == ShouldSample::No)
        return true;

    static const double selectionProbability = 0.05;
    return cryptographicallyRandomUnitInterval() <= selectionProbability;
}

} // namespace WebCore
