/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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

#include "ExceptionOr.h"
#include "GPUBufferMapState.h"
#include "GPUIntegralTypes.h"
#include "GPUMapMode.h"
#include "JSDOMPromiseDeferred.h"
#include "JSDOMPromiseDeferredForward.h"
#include "WebGPUBuffer.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <cstdint>
#include <optional>
#include <wtf/HashSet.h>
#include <wtf/Range.h>
#include <wtf/RangeSet.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUDevice;

class GPUBuffer : public RefCountedAndCanMakeWeakPtr<GPUBuffer> {
public:
    static Ref<GPUBuffer> create(Ref<WebGPU::Buffer>&& backing, size_t bufferSize, GPUBufferUsageFlags usage, bool mappedAtCreation, GPUDevice& device)
    {
        return adoptRef(*new GPUBuffer(WTFMove(backing), bufferSize, usage, mappedAtCreation, device));
    }

    String label() const;
    void setLabel(String&&);

    using MapAsyncPromise = DOMPromiseDeferred<IDLNull>;
    void mapAsync(GPUMapModeFlags, std::optional<GPUSize64> offset, std::optional<GPUSize64> sizeForMap, MapAsyncPromise&&);
    ExceptionOr<Ref<JSC::ArrayBuffer>> getMappedRange(std::optional<GPUSize64> offset, std::optional<GPUSize64> rangeSize);
    void unmap(ScriptExecutionContext&);

    void destroy(ScriptExecutionContext&);

    WebGPU::Buffer& backing() { return m_backing; }
    const WebGPU::Buffer& backing() const { return m_backing; }

    GPUSize64 size() const { return static_cast<GPUSize64>(m_bufferSize); }
    GPUBufferUsageFlags usage() const { return m_usage; }

    GPUBufferMapState mapState() const { return m_mapState; };

    ~GPUBuffer();
private:
    GPUBuffer(Ref<WebGPU::Buffer>&&, size_t, GPUBufferUsageFlags, bool, GPUDevice&);
    void internalUnmap(ScriptExecutionContext&);

    Ref<WebGPU::Buffer> m_backing;
    struct ArrayBufferWithOffset {
        RefPtr<JSC::ArrayBuffer> buffer;
        size_t offset { 0 };
    };
    Vector<ArrayBufferWithOffset> m_arrayBuffers;
    size_t m_bufferSize { 0 };
    size_t m_mappedRangeOffset { 0 };
    size_t m_mappedRangeSize { 0 };
    const GPUBufferUsageFlags m_usage { 0 };
    GPUBufferMapState m_mapState { GPUBufferMapState::Unmapped };
    std::optional<MapAsyncPromise> m_pendingMapPromise;
    WeakPtr<GPUDevice, WeakPtrImplWithEventTargetData> m_device;
    using MappedRanges = WTF::RangeSet<WTF::Range<size_t>>;
    MappedRanges m_mappedRanges;
    HashSet<size_t, DefaultHash<size_t>, WTF::UnsignedWithZeroKeyHashTraits<size_t>> m_mappedPoints;
    bool m_destroyed { false };
    bool m_mappedAtCreation { false };
};

}
