/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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

#include <wtf/Assertions.h>
#include <wtf/Vector.h>
#include <limits.h>

namespace JSC {
    template<typename Traits>
    class BytecodeGeneratorBase;
    struct JSGeneratorTraits;

    template<typename Traits>
    class GenericLabel;

    template<typename Traits>
    class GenericBoundLabel {
        using BytecodeGenerator = BytecodeGeneratorBase<Traits>;
        using Label = GenericLabel<Traits>;

    public:
        GenericBoundLabel()
            : m_type(Offset)
            , m_generator(nullptr)
            , m_target(0)
        { }

        explicit GenericBoundLabel(int offset)
            : m_type(Offset)
            , m_generator(nullptr)
            , m_target(offset)
        { }

        GenericBoundLabel(BytecodeGenerator* generator, Label* label)
            : m_type(GeneratorForward)
            , m_generator(generator)
            , m_label(label)
        { }

        GenericBoundLabel(BytecodeGenerator* generator, int offset)
            : m_type(GeneratorBackward)
            , m_generator(generator)
            , m_target(offset)
        { }

        int target()
        {
            switch (m_type) {
            case Offset:
                return m_target;
            case GeneratorBackward:
                return m_target - m_generator->m_writer.position();
            case GeneratorForward:
                return 0;
            default:
                RELEASE_ASSERT_NOT_REACHED();
            }
        }

        int saveTarget()
        {
            if (m_type == GeneratorForward) {
                m_savedTarget = m_generator->m_writer.position();
                return 0;
            }

            m_savedTarget = target();
            return m_savedTarget;
        }

        int commitTarget()
        {
            if (m_type == GeneratorForward) {
                m_label->m_unresolvedJumps.append(m_savedTarget);
                return 0;
            }

            return m_savedTarget;
        }

        operator int() { return target(); }

    private:
        enum Type : uint8_t {
            Offset,
            GeneratorForward,
            GeneratorBackward,
        };

        Type m_type;
        int m_savedTarget { 0 };
        BytecodeGenerator* m_generator;
        union {
            Label* m_label;
            int m_target;
        };
    };

    template<typename Traits>
    class GenericLabel {
        WTF_MAKE_NONCOPYABLE(GenericLabel);
        using BytecodeGenerator = BytecodeGeneratorBase<Traits>;
        using BoundLabel = GenericBoundLabel<Traits>;
        using JumpVector = Vector<int, 8>;

    public:
        GenericLabel() = default;

        void setLocation(BytecodeGenerator&, unsigned location);

        BoundLabel bind(BytecodeGenerator* generator)
        {
            m_bound = true;
            if (!isForward())
                return BoundLabel(generator, m_location);
            return BoundLabel(generator, this);
        }

        BoundLabel bind(unsigned offset)
        {
            m_bound = true;
            if (!isForward())
                return BoundLabel(m_location - offset);
            m_unresolvedJumps.append(offset);
            return BoundLabel();
        }

        BoundLabel bind()
        {
            ASSERT(!isForward());
            return bind(0u);
        }

        void ref() { ++m_refCount; }
        void deref()
        {
            --m_refCount;
            ASSERT(m_refCount >= 0);
        }
        int refCount() const { return m_refCount; }
        bool hasOneRef() const { return m_refCount == 1; }

        bool isForward() const { return m_location == invalidLocation; }
        
        bool isBound() const { return m_bound; }

        unsigned location() const
        {
            ASSERT(!isForward());
            m_bound = true;
            return m_location;
        };

        const JumpVector& unresolvedJumps() const { return  m_unresolvedJumps; }

    private:
        friend BoundLabel;

        static constexpr unsigned invalidLocation = UINT_MAX;

        int m_refCount { 0 };
        unsigned m_location { invalidLocation };
        mutable bool m_bound { false };
        mutable JumpVector m_unresolvedJumps;
    };

    using Label = GenericLabel<JSGeneratorTraits>;
    using BoundLabel = GenericBoundLabel<JSGeneratorTraits>;

    namespace Wasm {
    struct GeneratorTraits;
    using Label = GenericLabel<Wasm::GeneratorTraits>;
    }

    // This cannot be declared in the Wasm namespace as it conflicts with the
    // ruby Wasm namespace when referencing it from BytecodeList.rb
    using WasmBoundLabel = GenericBoundLabel<Wasm::GeneratorTraits>;

} // namespace JSC
