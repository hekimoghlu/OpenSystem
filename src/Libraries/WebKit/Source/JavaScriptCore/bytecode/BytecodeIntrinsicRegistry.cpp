/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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
#include "BytecodeIntrinsicRegistry.h"

#include "AbstractModuleRecord.h"
#include "BuiltinNames.h"
#include "BytecodeGenerator.h"
#include "GlobalObjectMethodTable.h"
#include "IdentifierInlines.h"
#include "IterationKind.h"
#include "JSArrayIterator.h"
#include "JSAsyncFromSyncIterator.h"
#include "JSAsyncGenerator.h"
#include "JSGenerator.h"
#include "JSGlobalObject.h"
#include "JSIteratorHelper.h"
#include "JSMapIterator.h"
#include "JSModuleLoader.h"
#include "JSPromise.h"
#include "JSRegExpStringIterator.h"
#include "JSSetIterator.h"
#include "JSStringIterator.h"
#include "JSWrapForValidIterator.h"
#include "LinkTimeConstant.h"
#include "Nodes.h"
#include "StrongInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BytecodeIntrinsicRegistry);

#define INITIALIZE_BYTECODE_INTRINSIC_NAMES_TO_SET(name) m_bytecodeIntrinsicMap.add(vm.propertyNames->builtinNames().name##PrivateName().impl(), Entry(&BytecodeIntrinsicNode::emit_intrinsic_##name));
#define INITIALIZE_BYTECODE_INTRINSIC_NAMES_TO_SET_FOR_LINK_TIME_CONSTANT(name, code) m_bytecodeIntrinsicMap.add(vm.propertyNames->builtinNames().name##PrivateName().impl(), JSC::LinkTimeConstant::name);

BytecodeIntrinsicRegistry::BytecodeIntrinsicRegistry(VM& vm)
    : m_vm(vm)
    , m_bytecodeIntrinsicMap()
{
    JSC_COMMON_BYTECODE_INTRINSIC_FUNCTIONS_EACH_NAME(INITIALIZE_BYTECODE_INTRINSIC_NAMES_TO_SET)
    JSC_COMMON_BYTECODE_INTRINSIC_CONSTANTS_EACH_NAME(INITIALIZE_BYTECODE_INTRINSIC_NAMES_TO_SET)
    JSC_FOREACH_LINK_TIME_CONSTANTS(INITIALIZE_BYTECODE_INTRINSIC_NAMES_TO_SET_FOR_LINK_TIME_CONSTANT)

    m_undefined.set(m_vm, jsUndefined());
    m_Infinity.set(m_vm, jsDoubleNumber(std::numeric_limits<double>::infinity()));
    m_iterationKindKey.set(m_vm, jsNumber(static_cast<unsigned>(IterationKind::Keys)));
    m_iterationKindValue.set(m_vm, jsNumber(static_cast<unsigned>(IterationKind::Values)));
    m_iterationKindEntries.set(m_vm, jsNumber(static_cast<unsigned>(IterationKind::Entries)));
    m_MAX_ARRAY_INDEX.set(m_vm, jsNumber(MAX_ARRAY_INDEX));
    m_MAX_STRING_LENGTH.set(m_vm, jsNumber(JSString::MaxLength));
    m_MAX_SAFE_INTEGER.set(m_vm, jsDoubleNumber(maxSafeInteger()));
    m_ModuleFetch.set(m_vm, jsNumber(static_cast<unsigned>(JSModuleLoader::Status::Fetch)));
    m_ModuleInstantiate.set(m_vm, jsNumber(static_cast<unsigned>(JSModuleLoader::Status::Instantiate)));
    m_ModuleSatisfy.set(m_vm, jsNumber(static_cast<unsigned>(JSModuleLoader::Status::Satisfy)));
    m_ModuleLink.set(m_vm, jsNumber(static_cast<unsigned>(JSModuleLoader::Status::Link)));
    m_ModuleReady.set(m_vm, jsNumber(static_cast<unsigned>(JSModuleLoader::Status::Ready)));
    m_promiseRejectionReject.set(m_vm, jsNumber(static_cast<unsigned>(JSPromiseRejectionOperation::Reject)));
    m_promiseRejectionHandle.set(m_vm, jsNumber(static_cast<unsigned>(JSPromiseRejectionOperation::Handle)));
    m_promiseStatePending.set(m_vm, jsNumber(static_cast<unsigned>(JSPromise::Status::Pending)));
    m_promiseStateFulfilled.set(m_vm, jsNumber(static_cast<unsigned>(JSPromise::Status::Fulfilled)));
    m_promiseStateRejected.set(m_vm, jsNumber(static_cast<unsigned>(JSPromise::Status::Rejected)));
    m_promiseStateMask.set(m_vm, jsNumber(JSPromise::stateMask));
    m_promiseFlagsIsHandled.set(m_vm, jsNumber(JSPromise::isHandledFlag));
    m_promiseFlagsIsFirstResolvingFunctionCalled.set(m_vm, jsNumber(JSPromise::isFirstResolvingFunctionCalledFlag));
    // FIXME: Clean up JSInternalObjectImpl field registry.
    // https://bugs.webkit.org/show_bug.cgi?id=201894
    m_promiseFieldFlags.set(m_vm, jsNumber(static_cast<unsigned>(JSPromise::Field::Flags)));
    m_promiseFieldReactionsOrResult.set(m_vm, jsNumber(static_cast<unsigned>(JSPromise::Field::ReactionsOrResult)));
    m_generatorFieldState.set(m_vm, jsNumber(static_cast<unsigned>(JSGenerator::Field::State)));
    m_generatorFieldNext.set(m_vm, jsNumber(static_cast<unsigned>(JSGenerator::Field::Next)));
    m_generatorFieldThis.set(m_vm, jsNumber(static_cast<unsigned>(JSGenerator::Field::This)));
    m_generatorFieldFrame.set(m_vm, jsNumber(static_cast<unsigned>(JSGenerator::Field::Frame)));
    m_generatorFieldContext.set(m_vm, jsNumber(static_cast<unsigned>(JSGenerator::Field::Context)));
    m_GeneratorResumeModeNormal.set(m_vm, jsNumber(static_cast<int32_t>(JSGenerator::ResumeMode::NormalMode)));
    m_GeneratorResumeModeThrow.set(m_vm, jsNumber(static_cast<int32_t>(JSGenerator::ResumeMode::ThrowMode)));
    m_GeneratorResumeModeReturn.set(m_vm, jsNumber(static_cast<int32_t>(JSGenerator::ResumeMode::ReturnMode)));
    m_GeneratorStateCompleted.set(m_vm, jsNumber(static_cast<int32_t>(JSGenerator::State::Completed)));
    m_GeneratorStateExecuting.set(m_vm, jsNumber(static_cast<int32_t>(JSGenerator::State::Executing)));
    m_GeneratorStateInit.set(m_vm, jsNumber(static_cast<int32_t>(JSGenerator::State::Init)));
    m_arrayIteratorFieldIteratedObject.set(m_vm, jsNumber(static_cast<int32_t>(JSArrayIterator::Field::IteratedObject)));
    m_arrayIteratorFieldIndex.set(m_vm, jsNumber(static_cast<int32_t>(JSArrayIterator::Field::Index)));
    m_arrayIteratorFieldKind.set(m_vm, jsNumber(static_cast<int32_t>(JSArrayIterator::Field::Kind)));

    m_mapIteratorFieldEntry.set(m_vm, jsNumber(static_cast<int32_t>(JSMapIterator::Field::Entry)));
    m_mapIteratorFieldIteratedObject.set(m_vm, jsNumber(static_cast<int32_t>(JSMapIterator::Field::IteratedObject)));
    m_mapIteratorFieldStorage.set(m_vm, jsNumber(static_cast<int32_t>(JSMapIterator::Field::Storage)));
    m_mapIteratorFieldKind.set(m_vm, jsNumber(static_cast<int32_t>(JSMapIterator::Field::Kind)));
    m_setIteratorFieldEntry.set(m_vm, jsNumber(static_cast<int32_t>(JSSetIterator::Field::Entry)));
    m_setIteratorFieldIteratedObject.set(m_vm, jsNumber(static_cast<int32_t>(JSSetIterator::Field::IteratedObject)));
    m_setIteratorFieldStorage.set(m_vm, jsNumber(static_cast<int32_t>(JSSetIterator::Field::Storage)));
    m_setIteratorFieldKind.set(m_vm, jsNumber(static_cast<int32_t>(JSSetIterator::Field::Kind)));
    m_stringIteratorFieldIndex.set(m_vm, jsNumber(static_cast<int32_t>(JSStringIterator::Field::Index)));
    m_stringIteratorFieldIteratedString.set(m_vm, jsNumber(static_cast<int32_t>(JSStringIterator::Field::IteratedString)));
    m_asyncGeneratorFieldSuspendReason.set(m_vm, jsNumber(static_cast<unsigned>(JSAsyncGenerator::Field::SuspendReason)));
    m_asyncGeneratorFieldQueueFirst.set(m_vm, jsNumber(static_cast<unsigned>(JSAsyncGenerator::Field::QueueFirst)));
    m_asyncGeneratorFieldQueueLast.set(m_vm, jsNumber(static_cast<unsigned>(JSAsyncGenerator::Field::QueueLast)));
    m_AsyncGeneratorStateCompleted.set(m_vm, jsNumber(static_cast<int32_t>(JSAsyncGenerator::AsyncGeneratorState::Completed)));
    m_AsyncGeneratorStateExecuting.set(m_vm, jsNumber(static_cast<int32_t>(JSAsyncGenerator::AsyncGeneratorState::Executing)));
    m_AsyncGeneratorStateSuspendedStart.set(m_vm, jsNumber(static_cast<int32_t>(JSAsyncGenerator::AsyncGeneratorState::SuspendedStart)));
    m_AsyncGeneratorStateSuspendedYield.set(m_vm, jsNumber(static_cast<int32_t>(JSAsyncGenerator::AsyncGeneratorState::SuspendedYield)));
    m_AsyncGeneratorStateAwaitingReturn.set(m_vm, jsNumber(static_cast<int32_t>(JSAsyncGenerator::AsyncGeneratorState::AwaitingReturn)));
    m_AsyncGeneratorSuspendReasonYield.set(m_vm, jsNumber(static_cast<int32_t>(JSAsyncGenerator::AsyncGeneratorSuspendReason::Yield)));
    m_AsyncGeneratorSuspendReasonAwait.set(m_vm, jsNumber(static_cast<int32_t>(JSAsyncGenerator::AsyncGeneratorSuspendReason::Await)));
    m_AsyncGeneratorSuspendReasonNone.set(m_vm, jsNumber(static_cast<int32_t>(JSAsyncGenerator::AsyncGeneratorSuspendReason::None)));
    m_asyncFromSyncIteratorFieldSyncIterator.set(m_vm, jsNumber(static_cast<int32_t>(JSAsyncFromSyncIterator::Field::SyncIterator)));
    m_asyncFromSyncIteratorFieldNextMethod.set(m_vm, jsNumber(static_cast<int32_t>(JSAsyncFromSyncIterator::Field::NextMethod)));
    m_abstractModuleRecordFieldState.set(m_vm, jsNumber(static_cast<int32_t>(AbstractModuleRecord::Field::State)));
    m_wrapForValidIteratorFieldIteratedIterator.set(m_vm, jsNumber(static_cast<int32_t>(JSWrapForValidIterator::Field::IteratedIterator)));
    m_wrapForValidIteratorFieldIteratedNextMethod.set(m_vm, jsNumber(static_cast<int32_t>(JSWrapForValidIterator::Field::IteratedNextMethod)));
    m_regExpStringIteratorFieldRegExp.set(m_vm, jsNumber(static_cast<int32_t>(JSRegExpStringIterator::Field::RegExp)));
    m_regExpStringIteratorFieldString.set(m_vm, jsNumber(static_cast<int32_t>(JSRegExpStringIterator::Field::String)));
    m_regExpStringIteratorFieldGlobal.set(m_vm, jsNumber(static_cast<int32_t>(JSRegExpStringIterator::Field::Global)));
    m_regExpStringIteratorFieldFullUnicode.set(m_vm, jsNumber(static_cast<int32_t>(JSRegExpStringIterator::Field::FullUnicode)));
    m_regExpStringIteratorFieldDone.set(m_vm, jsNumber(static_cast<int32_t>(JSRegExpStringIterator::Field::Done)));
    m_iteratorHelperFieldGenerator.set(m_vm, jsNumber(static_cast<int32_t>(JSIteratorHelper::Field::Generator)));
    m_iteratorHelperFieldUnderlyingIterator.set(m_vm, jsNumber(static_cast<int32_t>(JSIteratorHelper::Field::UnderlyingIterator)));
}

std::optional<BytecodeIntrinsicRegistry::Entry> BytecodeIntrinsicRegistry::lookup(const Identifier& ident) const
{
    if (!ident.isPrivateName())
        return std::nullopt;
    auto iterator = m_bytecodeIntrinsicMap.find(ident.impl());
    if (iterator == m_bytecodeIntrinsicMap.end())
        return std::nullopt;
    return iterator->value;
}

#define JSC_DECLARE_BYTECODE_INTRINSIC_CONSTANT_GENERATORS(name) \
    JSValue BytecodeIntrinsicRegistry::name##Value(BytecodeGenerator&) \
    { \
        return m_##name.get(); \
    }
    JSC_COMMON_BYTECODE_INTRINSIC_CONSTANTS_SIMPLE_EACH_NAME(JSC_DECLARE_BYTECODE_INTRINSIC_CONSTANT_GENERATORS)
#undef JSC_DECLARE_BYTECODE_INTRINSIC_CONSTANT_GENERATORS

JSValue BytecodeIntrinsicRegistry::orderedHashTableSentinelValue(BytecodeGenerator& generator)
{
    return generator.vm().orderedHashTableSentinel();
}

} // namespace JSC

