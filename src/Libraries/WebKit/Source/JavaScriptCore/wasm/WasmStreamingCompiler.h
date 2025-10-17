/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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

#include "WasmStreamingParser.h"

#if ENABLE(WEBASSEMBLY)

#include "DeferredWorkTimer.h"
#include "JSCJSValue.h"

namespace JSC {

class CallLinkInfo;
class JSGlobalObject;
class JSObject;
class JSPromise;
class VM;

namespace Wasm {

class EntryPlan;
class StreamingPlan;

class StreamingCompiler final : public StreamingParserClient, public ThreadSafeRefCounted<StreamingCompiler> {
public:
    JS_EXPORT_PRIVATE static Ref<StreamingCompiler> create(VM&, CompilerMode, JSGlobalObject*, JSPromise*, JSObject*);

    JS_EXPORT_PRIVATE ~StreamingCompiler();

    void addBytes(std::span<const uint8_t> bytes) { m_parser.addBytes(bytes); }
    JS_EXPORT_PRIVATE void finalize(JSGlobalObject*);
    JS_EXPORT_PRIVATE void fail(JSGlobalObject*, JSValue);
    JS_EXPORT_PRIVATE void cancel();

    void didCompileFunction(StreamingPlan&);

private:
    JS_EXPORT_PRIVATE StreamingCompiler(VM&, CompilerMode, JSGlobalObject*, JSPromise*, JSObject*);

    bool didReceiveFunctionData(FunctionCodeIndex, const FunctionData&) final;
    void didFinishParsing() final;
    void didComplete() WTF_REQUIRES_LOCK(m_lock);
    void completeIfNecessary() WTF_REQUIRES_LOCK(m_lock);

    VM& m_vm;
    CompilerMode m_compilerMode;
    bool m_eagerFailed WTF_GUARDED_BY_LOCK(m_lock) { false };
    bool m_finalized WTF_GUARDED_BY_LOCK(m_lock) { false };
    bool m_threadedCompilationStarted { false };
    Lock m_lock;
    unsigned m_remainingCompilationRequests { 0 };
    DeferredWorkTimer::Ticket m_ticket;
    Ref<Wasm::ModuleInformation> m_info;
    StreamingParser m_parser;
    RefPtr<EntryPlan> m_plan;
};


} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
