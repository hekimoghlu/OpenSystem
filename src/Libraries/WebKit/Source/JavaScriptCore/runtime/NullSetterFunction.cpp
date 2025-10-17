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
#include "NullSetterFunction.h"

#include "CodeBlock.h"
#include "JSCInlines.h"

namespace JSC {

const ClassInfo NullSetterFunction::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(NullSetterFunction) };


#if ASSERT_ENABLED

class GetCallerStrictnessFunctor {
public:
    GetCallerStrictnessFunctor()
        : m_iterations(0)
        , m_callerIsStrict(false)
    {
    }

    IterationStatus operator()(StackVisitor& visitor) const
    {
        ++m_iterations;
        if (m_iterations < 2)
            return IterationStatus::Continue;

        CodeBlock* codeBlock = visitor->codeBlock();
        // This does not take into account that we might have an strict opcode in a non-strict context, but that's
        // ok since we assert below that this function should never be called from any kind strict context.
        m_callerIsStrict = codeBlock && codeBlock->ownerExecutable()->isInStrictContext();
        return IterationStatus::Done;
    }

    bool callerIsStrict() const { return m_callerIsStrict; }

private:
    mutable int m_iterations;
    mutable bool m_callerIsStrict;
};

static bool callerIsStrict(VM& vm, CallFrame* callFrame)
{
    GetCallerStrictnessFunctor iter;
    StackVisitor::visit(callFrame, vm, iter);
    return iter.callerIsStrict();
}

#endif // ASSERT_ENABLED

namespace NullSetterFunctionInternal {

static JSC_DECLARE_HOST_FUNCTION(callReturnUndefined);
static JSC_DECLARE_HOST_FUNCTION(callThrowError);

JSC_DEFINE_HOST_FUNCTION(callReturnUndefined, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
#if !ASSERT_ENABLED
    UNUSED_PARAM(globalObject);
    UNUSED_PARAM(callFrame);
#endif
    ASSERT(!callerIsStrict(globalObject->vm(), callFrame));
    return JSValue::encode(jsUndefined());
}

JSC_DEFINE_HOST_FUNCTION(callThrowError, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // This function is only called from IC. And we do not want to include this frame in Error's stack.
    constexpr bool useCurrentFrame = false;
    throwException(globalObject, scope, ErrorInstance::create(vm, globalObject->errorStructure(ErrorType::TypeError), ReadonlyPropertyWriteError, JSValue(), nullptr, TypeNothing, ErrorType::TypeError, useCurrentFrame));
    return { };
}

}

NullSetterFunction::NullSetterFunction(VM& vm, Structure* structure, ECMAMode ecmaMode)
    : Base(vm, structure, ecmaMode.isStrict() ? NullSetterFunctionInternal::callThrowError : NullSetterFunctionInternal::callReturnUndefined, nullptr)
{
}

}
