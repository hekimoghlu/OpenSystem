/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#include <wtf/text/WTFString.h>

namespace WebCore {

class MockContentFilterSettings {
    friend class NeverDestroyed<MockContentFilterSettings>;
public:
    enum class DecisionPoint : uint8_t {
        AfterWillSendRequest,
        AfterRedirect,
        AfterResponse,
        AfterAddData,
        AfterFinishedAddingData,
        Never
    };

    enum class Decision : bool {
        Allow,
        Block
    };

    WEBCORE_TESTSUPPORT_EXPORT static MockContentFilterSettings& singleton();
    WEBCORE_TESTSUPPORT_EXPORT static void reset();
    static ASCIILiteral unblockURLHost() { return "mock-unblock"_s; }

    // Trick the generated bindings into thinking we're RefCounted.
    void ref() { }
    void deref() { }

    bool enabled() const { return m_enabled; }
    WEBCORE_TESTSUPPORT_EXPORT void setEnabled(bool);

    const String& blockedString() const { return m_blockedString; }
    WEBCORE_TESTSUPPORT_EXPORT void setBlockedString(const String&);

    DecisionPoint decisionPoint() const { return m_decisionPoint; }
    WEBCORE_TESTSUPPORT_EXPORT void setDecisionPoint(DecisionPoint);

    Decision decision() const { return m_decision; }
    WEBCORE_TESTSUPPORT_EXPORT void setDecision(Decision);

    Decision unblockRequestDecision() const { return m_unblockRequestDecision; }
    WEBCORE_TESTSUPPORT_EXPORT void setUnblockRequestDecision(Decision);

    WEBCORE_TESTSUPPORT_EXPORT const String& unblockRequestURL() const;

    const String& modifiedRequestURL() const { return m_modifiedRequestURL; }
    WEBCORE_TESTSUPPORT_EXPORT void setModifiedRequestURL(const String&);

    MockContentFilterSettings() = default;
    MockContentFilterSettings(const MockContentFilterSettings&) = default;
    MockContentFilterSettings& operator=(const MockContentFilterSettings&) = default;
private:
    friend struct IPC::ArgumentCoder<MockContentFilterSettings, void>;

    bool m_enabled { false };
    DecisionPoint m_decisionPoint { DecisionPoint::AfterResponse };
    Decision m_decision { Decision::Allow };
    Decision m_unblockRequestDecision { Decision::Block };
    String m_blockedString;
    String m_modifiedRequestURL;
};

} // namespace WebCore
