/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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

#include "Event.h"
#include "SecurityPolicyViolationEventDisposition.h"

namespace WebCore {

struct SecurityPolicyViolationEventInit : EventInit {
    SecurityPolicyViolationEventInit() = default;
    WEBCORE_EXPORT SecurityPolicyViolationEventInit(EventInit&&, String&& documentURI, String&& referrer, String&& blockedURI, String&& violatedDirective, String&& effectiveDirective, String&& originalPolicy, String&& sourceFile, String&& sample, SecurityPolicyViolationEventDisposition, unsigned short statusCode, unsigned lineNumber, unsigned columnNumber);

    String documentURI;
    String referrer;
    String blockedURI;
    String violatedDirective;
    String effectiveDirective;
    String originalPolicy;
    String sourceFile;
    String sample;
    SecurityPolicyViolationEventDisposition disposition { SecurityPolicyViolationEventDisposition::Enforce };
    unsigned short statusCode { 0 };
    unsigned lineNumber { 0 };
    unsigned columnNumber { 0 };
};

class SecurityPolicyViolationEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SecurityPolicyViolationEvent);
public:
    using Disposition = SecurityPolicyViolationEventDisposition;
    using Init = SecurityPolicyViolationEventInit;

    static Ref<SecurityPolicyViolationEvent> create(const AtomString& type, const Init& initializer = { }, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new SecurityPolicyViolationEvent(type, initializer, isTrusted));
    }

    const String& documentURI() const { return m_documentURI; }
    const String& referrer() const { return m_referrer; }
    const String& blockedURI() const { return m_blockedURI; }
    const String& violatedDirective() const { return m_violatedDirective; }
    const String& effectiveDirective() const { return m_effectiveDirective; }
    const String& originalPolicy() const { return m_originalPolicy; }
    const String& sourceFile() const { return m_sourceFile; }
    const String& sample() const { return m_sample; }
    Disposition disposition() const { return m_disposition; }
    unsigned short statusCode() const { return m_statusCode; }
    unsigned lineNumber() const { return m_lineNumber; }
    unsigned columnNumber() const { return m_columnNumber; }

private:
    SecurityPolicyViolationEvent()
        : Event(EventInterfaceType::SecurityPolicyViolationEvent)
    {
    }

    SecurityPolicyViolationEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
        : Event(EventInterfaceType::SecurityPolicyViolationEvent, type, initializer, isTrusted)
        , m_documentURI(initializer.documentURI)
        , m_referrer(initializer.referrer)
        , m_blockedURI(initializer.blockedURI)
        , m_violatedDirective(initializer.violatedDirective)
        , m_effectiveDirective(initializer.effectiveDirective)
        , m_originalPolicy(initializer.originalPolicy)
        , m_sourceFile(initializer.sourceFile)
        , m_sample(initializer.sample)
        , m_disposition(initializer.disposition)
        , m_statusCode(initializer.statusCode)
        , m_lineNumber(initializer.lineNumber)
        , m_columnNumber(initializer.columnNumber)
    {
    }

    String m_documentURI;
    String m_referrer;
    String m_blockedURI;
    String m_violatedDirective;
    String m_effectiveDirective;
    String m_originalPolicy;
    String m_sourceFile;
    String m_sample;
    Disposition m_disposition { Disposition::Enforce };
    unsigned short m_statusCode;
    unsigned m_lineNumber;
    unsigned m_columnNumber;
};

} // namespace WebCore
