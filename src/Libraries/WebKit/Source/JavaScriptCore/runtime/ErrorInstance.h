/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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

#include "ErrorType.h"
#include "JSObject.h"
#include "RuntimeType.h"
#include "StackFrame.h"

namespace JSC {

class CallLinkInfo;

class ErrorInstance : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnSpecialPropertyNames | OverridesPut | GetOwnPropertySlotIsImpureForPropertyAbsence;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<ErrorInstance*>(cell)->ErrorInstance::~ErrorInstance();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.errorInstanceSpace<mode>();
    }

    enum SourceTextWhereErrorOccurred { FoundExactSource, FoundApproximateSource };
    typedef String (*SourceAppender) (const String& originalMessage, StringView sourceText, RuntimeType, SourceTextWhereErrorOccurred);

    DECLARE_EXPORT_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static ErrorInstance* create(VM& vm, Structure* structure, const String& message, JSValue cause, SourceAppender appender = nullptr, RuntimeType type = TypeNothing, ErrorType errorType = ErrorType::Error, bool useCurrentFrame = true)
    {
        ErrorInstance* instance = new (NotNull, allocateCell<ErrorInstance>(vm)) ErrorInstance(vm, structure, errorType);
        instance->finishCreation(vm, message, cause, appender, type, useCurrentFrame);
        return instance;
    }

    static ErrorInstance* create(VM& vm, Structure* structure, const String& message, JSValue cause, ErrorType errorType, JSCell* owner, CallLinkInfo* callLinkInfo)
    {
        ErrorInstance* instance = new (NotNull, allocateCell<ErrorInstance>(vm)) ErrorInstance(vm, structure, errorType);
        instance->finishCreation(vm, message, cause, owner, callLinkInfo);
        return instance;
    }

    JS_EXPORT_PRIVATE static ErrorInstance* create(JSGlobalObject*, String&& message, ErrorType, LineColumn, String&& sourceURL, String&& stackString);
    static ErrorInstance* create(JSGlobalObject*, Structure*, JSValue message, JSValue options, SourceAppender = nullptr, RuntimeType = TypeNothing, ErrorType = ErrorType::Error, bool useCurrentFrame = true);

    bool hasSourceAppender() const { return !!m_sourceAppender; }
    SourceAppender sourceAppender() const { return m_sourceAppender; }
    void setSourceAppender(SourceAppender appender) { m_sourceAppender = appender; }
    void clearSourceAppender() { m_sourceAppender = nullptr; }
    void setRuntimeTypeForCause(RuntimeType type) { m_runtimeTypeForCause = type; }
    RuntimeType runtimeTypeForCause() const { return m_runtimeTypeForCause; }
    void clearRuntimeTypeForCause() { m_runtimeTypeForCause = TypeNothing; }

    ErrorType errorType() const { return m_errorType; }
    void setStackOverflowError()
    {
#if ENABLE(WEBASSEMBLY)
        m_catchableFromWasm = false;
#endif
        m_stackOverflowError = true;
    }
    bool isStackOverflowError() const { return m_stackOverflowError; }
    void setOutOfMemoryError()
    {
#if ENABLE(WEBASSEMBLY)
        m_catchableFromWasm = false;
#endif
        m_outOfMemoryError = true;
    }
    bool isOutOfMemoryError() const { return m_outOfMemoryError; }

    void setNativeGetterTypeError() { m_nativeGetterTypeError = true; }
    bool isNativeGetterTypeError() const { return m_nativeGetterTypeError; }

#if ENABLE(WEBASSEMBLY)
    void setCatchableFromWasm(bool flag) { m_catchableFromWasm = flag; }
    bool isCatchableFromWasm() const { return m_catchableFromWasm; }
#endif

    JS_EXPORT_PRIVATE String sanitizedToString(JSGlobalObject*);
    JS_EXPORT_PRIVATE String sanitizedMessageString(JSGlobalObject*);
    JS_EXPORT_PRIVATE String sanitizedNameString(JSGlobalObject*);
    
    JS_EXPORT_PRIVATE String tryGetMessageForDebugging();

    Vector<StackFrame>* stackTrace() { return m_stackTrace.get(); }

    bool materializeErrorInfoIfNeeded(VM&);
    bool materializeErrorInfoIfNeeded(VM&, PropertyName);

    void finalizeUnconditionally(VM&, CollectionScope);

protected:
    explicit ErrorInstance(VM&, Structure*, ErrorType);

    void finishCreation(VM&, const String& message, JSValue cause, SourceAppender = nullptr, RuntimeType = TypeNothing, bool useCurrentFrame = true);
    void finishCreation(VM&, const String& message, JSValue cause, JSCell* owner, CallLinkInfo*);
    void finishCreation(VM&, String&& message, LineColumn, String&& sourceURL, String&& stackString);

    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static void getOwnSpecialPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    static bool defineOwnProperty(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool shouldThrow);
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);

    void computeErrorInfo(VM&);

    SourceAppender m_sourceAppender { nullptr };
    std::unique_ptr<Vector<StackFrame>> m_stackTrace;
    LineColumn m_lineColumn;
    String m_sourceURL;
    String m_stackString;
    RuntimeType m_runtimeTypeForCause { TypeNothing };
    ErrorType m_errorType { ErrorType::Error };
    bool m_stackOverflowError : 1;
    bool m_outOfMemoryError : 1;
    bool m_errorInfoMaterialized : 1;
    bool m_nativeGetterTypeError : 1;
#if ENABLE(WEBASSEMBLY)
    bool m_catchableFromWasm : 1;
#endif
};

String appendSourceToErrorMessage(CodeBlock*, BytecodeIndex, const String&, RuntimeType, ErrorInstance::SourceAppender);

} // namespace JSC
