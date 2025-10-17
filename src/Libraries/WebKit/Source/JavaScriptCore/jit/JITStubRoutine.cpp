/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
#include "config.h"
#include "JITStubRoutine.h"

#include "AccessCase.h"
#include "CallLinkInfo.h"
#include "GCAwareJITStubRoutine.h"
#include "PolymorphicCallStubRoutine.h"

namespace JSC {

void JITStubRoutine::observeZeroRefCountImpl()
{
    RELEASE_ASSERT(!m_refCount);
    delete this;
}

template<typename Func>
void JITStubRoutine::runWithDowncast(const Func& function)
{
    switch (m_type) {
    case Type::JITStubRoutineType:
        function(static_cast<JITStubRoutine*>(this));
        break;
    case Type::GCAwareJITStubRoutineType:
        function(static_cast<GCAwareJITStubRoutine*>(this));
        break;
    case Type::PolymorphicCallStubRoutineType:
        function(static_cast<PolymorphicCallStubRoutine*>(this));
        break;
#if ENABLE(JIT)
    case Type::PolymorphicAccessJITStubRoutineType:
        function(static_cast<PolymorphicAccessJITStubRoutine*>(this));
        break;
    case Type::MarkingGCAwareJITStubRoutineType:
        function(static_cast<MarkingGCAwareJITStubRoutine*>(this));
        break;
    case Type::GCAwareJITStubRoutineWithExceptionHandlerType:
        function(static_cast<GCAwareJITStubRoutineWithExceptionHandler*>(this));
        break;
#endif
    }
}

void JITStubRoutine::aboutToDie()
{
    runWithDowncast([&](auto* derived) {
        derived->aboutToDieImpl();
    });
}

void JITStubRoutine::observeZeroRefCount()
{
    runWithDowncast([&](auto* derived) {
        derived->observeZeroRefCountImpl();
    });
}

bool JITStubRoutine::visitWeak(VM& vm)
{
    bool result = true;
    runWithDowncast([&](auto* derived) {
        result = derived->visitWeakImpl(vm);
    });
    return result;
}

CallLinkInfo* JITStubRoutine::callLinkInfoAt(const ConcurrentJSLocker& locker, unsigned index)
{
    CallLinkInfo* result = nullptr;
    runWithDowncast([&](auto* derived) {
        result = derived->callLinkInfoAtImpl(locker, index);
    });
    return result;
}

void JITStubRoutine::markRequiredObjects(AbstractSlotVisitor& visitor)
{
    runWithDowncast([&](auto* derived) {
        derived->markRequiredObjectsImpl(visitor);
    });
}

void JITStubRoutine::markRequiredObjects(SlotVisitor& visitor)
{
    runWithDowncast([&](auto* derived) {
        derived->markRequiredObjectsImpl(visitor);
    });
}

void JITStubRoutine::operator delete(JITStubRoutine* stubRoutine, std::destroying_delete_t)
{
    stubRoutine->runWithDowncast([&](auto* derived) {
        std::decay_t<decltype(*derived)>::destroy(derived);
    });
}

} // namespace JSC
