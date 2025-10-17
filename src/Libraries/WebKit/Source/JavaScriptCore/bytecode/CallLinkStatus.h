/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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

#include "CallLinkInfo.h"
#include "CallVariant.h"
#include "CodeOrigin.h"
#include "ConcurrentJSLock.h"
#include "ExitFlag.h"
#include "ICStatusMap.h"
#include "JSCJSValue.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

class CodeBlock;
class InternalFunction;
class JSFunction;
class Structure;
class CallLinkInfo;

class CallLinkStatus final {
    WTF_MAKE_TZONE_ALLOCATED(CallLinkStatus);
public:
    CallLinkStatus()
    {
    }
    
    static CallLinkStatus takesSlowPath()
    {
        CallLinkStatus result;
        result.m_couldTakeSlowPath = true;
        return result;
    }
    
    explicit CallLinkStatus(JSValue);
    
    CallLinkStatus(CallVariant variant)
        : m_variants(1, variant)
    {
    }
    
    struct ExitSiteData {
        ExitFlag takesSlowPath;
        ExitFlag badFunction;
    };
    static ExitSiteData computeExitSiteData(CodeBlock*, BytecodeIndex);
    
    static CallLinkStatus computeFor(CodeBlock*, BytecodeIndex, const ICStatusMap&, ExitSiteData);
    static CallLinkStatus computeFor(CodeBlock*, BytecodeIndex, const ICStatusMap&);

#if ENABLE(JIT)
    // Computes the status assuming that we never took slow path and never previously
    // exited.
    static CallLinkStatus computeFor(const ConcurrentJSLocker&, CodeBlock*, CallLinkInfo&);
    
    // Computes the status accounting for exits.
    static CallLinkStatus computeFor(
        const ConcurrentJSLocker&, CodeBlock*, CallLinkInfo&, ExitSiteData, ExitingInlineKind = ExitFromAnyInlineKind);
#endif
    
    static CallLinkStatus computeFor(
        CodeBlock*, CodeOrigin, const ICStatusMap&, const ICStatusContextStack&);
    
    void setProvenConstantCallee(CallVariant);
    
    bool isSet() const { return !m_variants.isEmpty() || m_couldTakeSlowPath; }
    
    explicit operator bool() const { return isSet(); }
    
    bool couldTakeSlowPath() const { return m_couldTakeSlowPath; }
    
    void setCouldTakeSlowPath(bool value) { m_couldTakeSlowPath = value; }
    
    CallVariantList variants() const { return m_variants; }
    unsigned size() const { return m_variants.size(); }
    CallVariant at(unsigned i) const { return m_variants[i]; }
    CallVariant operator[](unsigned i) const { return at(i); }
    bool isProved() const { return m_isProved; }
    bool isBasedOnStub() const { return m_isBasedOnStub; }
    bool canOptimize() const { return !m_variants.isEmpty(); }

    bool isClosureCall() const; // Returns true if any callee is a closure call.
    
    unsigned maxArgumentCountIncludingThisForVarargs() const { return m_maxArgumentCountIncludingThisForVarargs; }
    
    bool finalize(VM&);
    
    void merge(const CallLinkStatus&);
    
    void filter(JSValue);
    
    void dump(PrintStream&) const;
    
private:
    void makeClosureCall();
    
#if ENABLE(JIT)
    static CallLinkStatus computeFromCallLinkInfo(
        const ConcurrentJSLocker&, CallLinkInfo&);
#endif
    
    void accountForExits(ExitSiteData, ExitingInlineKind);
    
    CallVariantList m_variants;
    bool m_couldTakeSlowPath { false };
    bool m_isProved { false };
    bool m_isBasedOnStub { false };
    uint8_t m_maxArgumentCountIncludingThisForVarargs { 0 }; // More than UINT8_MAX will be recorded as UINT8_MAX.
};

} // namespace JSC
