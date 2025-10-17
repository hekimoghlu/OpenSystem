/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

#include "CacheableIdentifierInlines.h"
#include "ClassInfo.h"
#include "Identifier.h"
#include <wtf/Condition.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/PrintStream.h>
#include <wtf/Spectrum.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

#define FOR_EACH_ICEVENT_KIND(macro) \
    macro(InvalidKind) \
    macro(GetByAddAccessCase) \
    macro(GetByReplaceWithJump) \
    macro(GetBySelfPatch) \
    macro(InAddAccessCase) \
    macro(InReplaceWithJump) \
    macro(InReplaceWithGeneric) \
    macro(InstanceOfAddAccessCase) \
    macro(InstanceOfReplaceWithJump) \
    macro(OperationGetById) \
    macro(OperationGetByIdGeneric) \
    macro(OperationGetByIdBuildList) \
    macro(OperationGetByIdOptimize) \
    macro(OperationGetByValOptimize) \
    macro(OperationGetByIdWithThisOptimize) \
    macro(OperationGetByValWithThisOptimize) \
    macro(OperationGenericIn) \
    macro(OperationInByIdGeneric) \
    macro(OperationInByIdOptimize) \
    macro(OperationPutByIdStrict) \
    macro(OperationPutByIdSloppy) \
    macro(OperationPutByIdDirectStrict) \
    macro(OperationPutByIdDirectSloppy) \
    macro(OperationPutByIdStrictOptimize) \
    macro(OperationPutByIdSloppyOptimize) \
    macro(OperationPutByIdDirectStrictOptimize) \
    macro(OperationPutByIdDirectSloppyOptimize) \
    macro(OperationPutByIdStrictBuildList) \
    macro(OperationPutByIdSloppyBuildList) \
    macro(OperationPutByIdDefinePrivateFieldStrictOptimize) \
    macro(OperationPutByIdPutPrivateFieldStrictOptimize) \
    macro(PutByAddAccessCase) \
    macro(PutByReplaceWithJump) \
    macro(PutBySelfPatch) \
    macro(InBySelfPatch) \
    macro(DelByReplaceWithJump) \
    macro(DelByReplaceWithGeneric) \
    macro(OperationGetPrivateNameOptimize) \
    macro(OperationGetPrivateNameById) \
    macro(OperationGetPrivateNameByIdOptimize) \
    macro(OperationGetPrivateNameByIdGeneric) \
    macro(CheckPrivateBrandAddAccessCase) \
    macro(SetPrivateBrandAddAccessCase) \
    macro(CheckPrivateBrandReplaceWithJump) \
    macro(SetPrivateBrandReplaceWithJump) \
    macro(OperationPutByIdSetPrivateFieldStrictOptimize)

class ICEvent {
public:
    enum Kind {
#define ICEVENT_KIND_DECLARATION(name) name,
        FOR_EACH_ICEVENT_KIND(ICEVENT_KIND_DECLARATION)
#undef ICEVENT_KIND_DECLARATION
    };

    enum PropertyLocation {
        Unknown,
        BaseObject,
        ProtoLookup
    };

    ICEvent()
    {
    }
    
    ICEvent(VM& vm, Kind kind, const ClassInfo* classInfo, PropertyName propertyName)
        : m_kind(kind)
        , m_classInfo(classInfo)
        , m_propertyName(Identifier::fromUid(vm, propertyName.uid()))
        , m_propertyLocation(Unknown)
    {
    }

    ICEvent(VM& vm, Kind kind, const ClassInfo* classInfo, PropertyName propertyName, bool isBaseProperty)
        : m_kind(kind)
        , m_classInfo(classInfo)
        , m_propertyName(Identifier::fromUid(vm, propertyName.uid()))
        , m_propertyLocation(isBaseProperty ? BaseObject : ProtoLookup)
    {
    }
    
    ICEvent(WTF::HashTableDeletedValueType)
        : m_kind(OperationGetById)
    {
    }
    
    bool operator==(const ICEvent& other) const
    {
        return m_kind == other.m_kind
            && m_classInfo == other.m_classInfo
            && m_propertyName == other.m_propertyName;
    }
    
    bool operator<(const ICEvent& other) const;
    bool operator>(const ICEvent& other) const { return other < *this; }
    bool operator<=(const ICEvent& other) const { return !(*this > other); }
    bool operator>=(const ICEvent& other) const { return !(*this < other); }
    
    explicit operator bool() const
    {
        return *this != ICEvent();
    }
    
    Kind kind() const { return m_kind; }
    const ClassInfo* classInfo() const { return m_classInfo; }
    const Identifier& propertyName() const { return m_propertyName; }
    
    unsigned hash() const
    {
        if (m_propertyName.isNull())
            return static_cast<unsigned>(m_kind) + static_cast<unsigned>(m_propertyLocation) + WTF::PtrHash<const ClassInfo*>::hash(m_classInfo);
        return static_cast<unsigned>(m_kind) + static_cast<unsigned>(m_propertyLocation) + WTF::PtrHash<const ClassInfo*>::hash(m_classInfo) + StringHash::hash(m_propertyName.string());
    }
    
    bool isHashTableDeletedValue() const
    {
        return *this == ICEvent(WTF::HashTableDeletedValue);
    }
    
    void dump(PrintStream&) const;
    
    void log() const;
    
private:
    
    Kind m_kind { InvalidKind };
    const ClassInfo* m_classInfo { nullptr };
    Identifier m_propertyName;
    PropertyLocation m_propertyLocation;
};

struct ICEventHash {
    static unsigned hash(const ICEvent& key) { return key.hash(); }
    static bool equal(const ICEvent& a, const ICEvent& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} // namespace JSC

namespace WTF {

void printInternal(PrintStream&, JSC::ICEvent::Kind);

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::ICEvent> : JSC::ICEventHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::ICEvent> : SimpleClassHashTraits<JSC::ICEvent> {
    static constexpr bool emptyValueIsZero = false;
};

} // namespace WTF

namespace JSC {

class ICStats {
    WTF_MAKE_NONCOPYABLE(ICStats);
    WTF_MAKE_TZONE_ALLOCATED(ICStats);
public:
    ICStats();
    ~ICStats();
    
    void add(const ICEvent& event);
    
    static ICStats& singleton();
    
private:

    Spectrum<ICEvent, uint64_t> m_spectrum;
    RefPtr<Thread> m_thread;
    Lock m_lock;
    Condition m_condition;
    bool m_shouldStop { false };
    
    static Atomic<ICStats*> s_instance;
};

#define LOG_IC(arguments) do {                  \
        if (Options::useICStats())              \
            (ICEvent arguments).log();          \
    } while (false)

} // namespace JSC
