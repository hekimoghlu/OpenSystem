/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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

#if ENABLE(DFG_JIT)

#include "ConcurrentJSLock.h"
#include "ExitKind.h"
#include "ExitingInlineKind.h"
#include "ExitingJITType.h"
#include <wtf/HashSet.h>
#include <wtf/Vector.h>

namespace JSC {

class UnlinkedCodeBlock;

namespace DFG {

class FrequentExitSite {
public:
    FrequentExitSite()
        : m_bytecodeIndex(BytecodeIndex(0))
        , m_kind(ExitKindUnset)
        , m_jitType(ExitFromAnything)
        , m_inlineKind(ExitFromAnyInlineKind)
    {
    }
    
    FrequentExitSite(WTF::HashTableDeletedValueType)
        : m_bytecodeIndex(WTF::HashTableDeletedValue)
        , m_kind(ExitKindUnset)
        , m_jitType(ExitFromAnything)
        , m_inlineKind(ExitFromAnyInlineKind)
    {
    }
    
    explicit FrequentExitSite(BytecodeIndex bytecodeIndex, ExitKind kind, ExitingJITType jitType = ExitFromAnything, ExitingInlineKind inlineKind = ExitFromAnyInlineKind)
        : m_bytecodeIndex(bytecodeIndex)
        , m_kind(kind)
        , m_jitType(jitType)
        , m_inlineKind(inlineKind)
    {
        if (m_kind == ArgumentsEscaped) {
            // Count this one globally. It doesn't matter where in the code block the arguments excaped;
            // the fact that they did is not associated with any particular instruction.
            m_bytecodeIndex = BytecodeIndex(0);
        }
    }
    
    // Use this constructor if you wish for the exit site to be counted globally within its
    // code block.
    explicit FrequentExitSite(ExitKind kind, ExitingJITType jitType = ExitFromAnything, ExitingInlineKind inlineKind = ExitFromAnyInlineKind)
        : m_bytecodeIndex(BytecodeIndex(0))
        , m_kind(kind)
        , m_jitType(jitType)
        , m_inlineKind(inlineKind)
    {
    }
    
    bool operator!() const
    {
        return m_kind == ExitKindUnset;
    }
    
    friend bool operator==(const FrequentExitSite&, const FrequentExitSite&) = default;
    
    bool subsumes(const FrequentExitSite& other) const
    {
        if (m_bytecodeIndex != other.m_bytecodeIndex)
            return false;
        if (m_kind != other.m_kind)
            return false;
        if (m_jitType != ExitFromAnything
            && m_jitType != other.m_jitType)
            return false;
        if (m_inlineKind != ExitFromAnyInlineKind
            && m_inlineKind != other.m_inlineKind)
            return false;
        return true;
    }
    
    unsigned hash() const
    {
        return m_bytecodeIndex.hash() + m_kind + static_cast<unsigned>(m_jitType) * 7 + static_cast<unsigned>(m_inlineKind) * 11;
    }
    
    BytecodeIndex bytecodeIndex() const { return m_bytecodeIndex; }
    ExitKind kind() const { return m_kind; }
    ExitingJITType jitType() const { return m_jitType; }
    ExitingInlineKind inlineKind() const { return m_inlineKind; }
    
    FrequentExitSite withJITType(ExitingJITType jitType) const
    {
        FrequentExitSite result = *this;
        result.m_jitType = jitType;
        return result;
    }

    FrequentExitSite withInlineKind(ExitingInlineKind inlineKind) const
    {
        FrequentExitSite result = *this;
        result.m_inlineKind = inlineKind;
        return result;
    }

    bool isHashTableDeletedValue() const
    {
        return m_kind == ExitKindUnset && m_bytecodeIndex.isHashTableDeletedValue();
    }
    
    void dump(PrintStream& out) const;

private:
    BytecodeIndex m_bytecodeIndex;
    ExitKind m_kind;
    ExitingJITType m_jitType;
    ExitingInlineKind m_inlineKind;
};

struct FrequentExitSiteHash {
    static unsigned hash(const FrequentExitSite& key) { return key.hash(); }
    static bool equal(const FrequentExitSite& a, const FrequentExitSite& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} } // namespace JSC::DFG


namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::DFG::FrequentExitSite> : JSC::DFG::FrequentExitSiteHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::DFG::FrequentExitSite> : SimpleClassHashTraits<JSC::DFG::FrequentExitSite> { };

} // namespace WTF

namespace JSC { namespace DFG {

class QueryableExitProfile;

class ExitProfile {
public:
    ExitProfile();
    ~ExitProfile();
    
    // Add a new frequent exit site. Return true if this is a new one, or false
    // if we already knew about it. This is an O(n) operation, because it errs
    // on the side of keeping the data structure compact. Also, this will only
    // be called a fixed number of times per recompilation. Recompilation is
    // rare to begin with, and implies doing O(n) operations on the CodeBlock
    // anyway.
    static bool add(CodeBlock*, const FrequentExitSite&);
    
    // Get the frequent exit sites for a bytecode index. This is O(n), and is
    // meant to only be used from debugging/profiling code.
    Vector<FrequentExitSite> exitSitesFor(BytecodeIndex);
    
    // This is O(n) and should be called on less-frequently executed code paths
    // in the compiler. It should be strictly cheaper than building a
    // QueryableExitProfile, if you really expect this to be called infrequently
    // and you believe that there are few exit sites.
    bool hasExitSite(const ConcurrentJSLocker&, const FrequentExitSite&) const;
    bool hasExitSite(const ConcurrentJSLocker& locker, ExitKind kind) const
    {
        return hasExitSite(locker, FrequentExitSite(kind));
    }
    
private:
    friend class QueryableExitProfile;
    
    std::unique_ptr<Vector<FrequentExitSite>> m_frequentExitSites;
};

class QueryableExitProfile {
public:
    QueryableExitProfile();
    ~QueryableExitProfile();
    
    void initialize(UnlinkedCodeBlock*);

    bool hasExitSite(const FrequentExitSite& site) const
    {
        if (site.jitType() == ExitFromAnything) {
            return hasExitSiteWithSpecificJITType(site.withJITType(ExitFromDFG))
                || hasExitSiteWithSpecificJITType(site.withJITType(ExitFromFTL));
        }
        return hasExitSiteWithSpecificJITType(site);
    }
    
    bool hasExitSite(ExitKind kind) const
    {
        return hasExitSite(FrequentExitSite(kind));
    }
    
    bool hasExitSite(BytecodeIndex bytecodeIndex, ExitKind kind) const
    {
        return hasExitSite(FrequentExitSite(bytecodeIndex, kind));
    }
private:
    bool hasExitSiteWithSpecificJITType(const FrequentExitSite& site) const
    {
        if (site.inlineKind() == ExitFromAnyInlineKind) {
            return hasExitSiteWithSpecificInlineKind(site.withInlineKind(ExitFromNotInlined))
                || hasExitSiteWithSpecificInlineKind(site.withInlineKind(ExitFromInlined));
        }
        return hasExitSiteWithSpecificInlineKind(site);
    }
    
    bool hasExitSiteWithSpecificInlineKind(const FrequentExitSite& site) const
    {
        return m_frequentExitSites.find(site) != m_frequentExitSites.end();
    }
    
    UncheckedKeyHashSet<FrequentExitSite> m_frequentExitSites;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
