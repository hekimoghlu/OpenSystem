/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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

#include "ConcurrentJSLock.h"
#include "SpeculatedType.h"
#include "Structure.h"
#include "VirtualRegister.h"
#include <span>
#include <wtf/PrintStream.h>
#include <wtf/StringPrintStream.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class UnlinkedValueProfile;

template<unsigned numberOfBucketsArgument, unsigned numberOfSpecFailBucketsArgument>
struct ValueProfileBase {
    friend class UnlinkedValueProfile;

    static constexpr unsigned numberOfBuckets = numberOfBucketsArgument;
    static constexpr unsigned numberOfSpecFailBuckets = numberOfSpecFailBucketsArgument;
    static constexpr unsigned totalNumberOfBuckets = numberOfBuckets + numberOfSpecFailBuckets;
    
    ValueProfileBase()
    {
        clearBuckets();
    }
    
    EncodedJSValue* specFailBucket(unsigned i)
    {
        ASSERT(numberOfBuckets + i < totalNumberOfBuckets);
        return m_buckets + numberOfBuckets + i;
    }

    void clearBuckets()
    {
        for (unsigned i = 0; i < totalNumberOfBuckets; ++i)
            clearEncodedJSValueConcurrent(m_buckets[i]);
    }
    
    const ClassInfo* classInfo(unsigned bucket) const
    {
        JSValue value = JSValue::decodeConcurrent(&m_buckets[bucket]);
        if (!!value) {
            if (!value.isCell())
                return nullptr;
            return value.asCell()->classInfo();
        }
        return nullptr;
    }
    
    unsigned numberOfSamples() const
    {
        unsigned result = 0;
        for (unsigned i = 0; i < totalNumberOfBuckets; ++i) {
            if (!!JSValue::decodeConcurrent(&m_buckets[i]))
                result++;
        }
        return result;
    }
    
    unsigned totalNumberOfSamples() const
    {
        return numberOfSamples() + isSampledBefore();
    }

    bool isSampledBefore() const { return m_prediction != SpecNone; }
    
    bool isLive() const
    {
        for (unsigned i = 0; i < totalNumberOfBuckets; ++i) {
            if (!!JSValue::decodeConcurrent(&m_buckets[i]))
                return true;
        }
        return false;
    }
    
    CString briefDescription(const ConcurrentJSLocker& locker)
    {
        SpeculatedType prediction = computeUpdatedPrediction(locker);
        
        StringPrintStream out;
        out.print("predicting ", SpeculationDump(prediction));
        return out.toCString();
    }
    
    void dump(PrintStream& out)
    {
        out.print("sampled before = ", isSampledBefore(), " live samples = ", numberOfSamples(), " prediction = ", SpeculationDump(m_prediction));
        bool first = true;
        for (unsigned i = 0; i < totalNumberOfBuckets; ++i) {
            JSValue value = JSValue::decode(m_buckets[i]);
            if (!!value) {
                if (first) {
                    out.printf(": ");
                    first = false;
                } else
                    out.printf(", ");
                out.print(value);
            }
        }
    }
    
    SpeculatedType computeUpdatedPrediction(const ConcurrentJSLocker&)
    {
        SpeculatedType merged = SpecNone;
        for (unsigned i = 0; i < totalNumberOfBuckets; ++i) {
            JSValue value = JSValue::decodeConcurrent(&m_buckets[i]);
            if (!value)
                continue;
            
            mergeSpeculation(merged, speculationFromValue(value));

            updateEncodedJSValueConcurrent(m_buckets[i], JSValue::encode(JSValue()));
        }

        mergeSpeculation(m_prediction, merged);
        
        return m_prediction;
    }

    void computeUpdatedPredictionForExtraValue(const ConcurrentJSLocker&, JSValue& value)
    {
        if (value)
            mergeSpeculation(m_prediction, speculationFromValue(value));
        value = JSValue();
    }

    EncodedJSValue m_buckets[totalNumberOfBuckets];

    SpeculatedType m_prediction { SpecNone };
};

struct MinimalValueProfile : public ValueProfileBase<0, 1> {
    MinimalValueProfile(): ValueProfileBase<0, 1>() { }
};

struct ValueProfile : public ValueProfileBase<1, 0> {
    ValueProfile() : ValueProfileBase<1, 0>() { }
    static constexpr ptrdiff_t offsetOfFirstBucket() { return OBJECT_OFFSETOF(ValueProfile, m_buckets[0]); }
};

struct ArgumentValueProfile : public ValueProfileBase<1, 1> {
    ArgumentValueProfile() : ValueProfileBase<1, 1>() { }
    static constexpr ptrdiff_t offsetOfFirstBucket() { return OBJECT_OFFSETOF(ValueProfile, m_buckets[0]); }
};

struct ValueProfileAndVirtualRegister : public ValueProfile {
    VirtualRegister m_operand;
};

static_assert(sizeof(ValueProfileAndVirtualRegister) >= sizeof(unsigned));
class alignas(ValueProfileAndVirtualRegister) ValueProfileAndVirtualRegisterBuffer final {
    WTF_MAKE_NONCOPYABLE(ValueProfileAndVirtualRegisterBuffer);
public:

    static ValueProfileAndVirtualRegisterBuffer* create(unsigned size)
    {
        void* buffer = VMMalloc::malloc(sizeof(ValueProfileAndVirtualRegisterBuffer) + size * sizeof(ValueProfileAndVirtualRegister));
        return new (buffer) ValueProfileAndVirtualRegisterBuffer(size);
    }

    static void destroy(ValueProfileAndVirtualRegisterBuffer* buffer)
    {
        buffer->~ValueProfileAndVirtualRegisterBuffer();
        VMMalloc::free(buffer);
    }

    template <typename Function>
    void forEach(Function function)
    {
        for (unsigned i = 0; i < m_size; ++i)
            function(data()[i]);
    }

    unsigned size() const { return m_size; }
    ValueProfileAndVirtualRegister* data() const
    {
        return std::bit_cast<ValueProfileAndVirtualRegister*>(this + 1);
    }

    std::span<ValueProfileAndVirtualRegister> span() { return { data(), size() }; }

private:

    ValueProfileAndVirtualRegisterBuffer(unsigned size)
        : m_size(size)
    {
        // FIXME: ValueProfile has more stuff than we need. We could optimize these value profiles
        // to be more space efficient.
        // https://bugs.webkit.org/show_bug.cgi?id=175413
        for (unsigned i = 0; i < m_size; ++i)
            new (&data()[i]) ValueProfileAndVirtualRegister();
    }

    ~ValueProfileAndVirtualRegisterBuffer()
    {
        for (unsigned i = 0; i < m_size; ++i)
            data()[i].~ValueProfileAndVirtualRegister();
    }

    unsigned m_size;
};

class UnlinkedValueProfile {
public:
    UnlinkedValueProfile() = default;

    void update(ValueProfile& profile)
    {
        SpeculatedType newType = profile.m_prediction | m_prediction;
        profile.m_prediction = newType;
        m_prediction = newType;
    }

    void update(ArgumentValueProfile& profile)
    {
        SpeculatedType newType = profile.m_prediction | m_prediction;
        profile.m_prediction = newType;
        m_prediction = newType;
    }

private:
    SpeculatedType m_prediction { SpecNone };
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
