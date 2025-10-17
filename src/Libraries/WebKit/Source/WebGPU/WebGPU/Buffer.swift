/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
import WebGPU_Internal

extension WebGPU.Buffer {
    var bufferContents: UnsafeMutableRawBufferPointer {
        UnsafeMutableRawBufferPointer(start: m_buffer.contents(), count: m_buffer.length)
    }

    func copy(from data: SpanConstUInt8, offset: Int) {
        let slice = bufferContents[offset...]
        // copyBytes(from:) checks bounds in debug builds only.
        // FIXME: Use a bounds-checking implementation when one is available.
        precondition(slice.count >= data.size_bytes())
        slice.copyBytes(from: data)
    }
}

// FIXME(emw): Find a way to generate thunks like these, maybe via a macro?
@_expose(Cxx)
public func Buffer_copyFrom_thunk(_ buffer: WebGPU.Buffer, from data: SpanConstUInt8, offset: Int) {
    buffer.copy(from: data, offset: offset)
}

@_expose(Cxx)
public func Buffer_getMappedRange_thunk(_ buffer: WebGPU.Buffer, offset: Int, size: Int) -> SpanUInt8 {
    return buffer.getMappedRange(offset: offset, size: size)
}

internal func computeRangeSize(size: Int, offset: Int) -> Int
{
    let result = checkedDifferenceSizeT(size, offset)
    if result.hasOverflowed() {
        return 0
    }
    return result.value()
}

extension WebGPU.Buffer {
    public func getMappedRange(offset: Int, size: Int) -> SpanUInt8
    {
        if !isValid() {
            return SpanUInt8()
        }

        var rangeSize = size
        if size == WGPU_WHOLE_MAP_SIZE {
            rangeSize = computeRangeSize(size: Int(currentSize()), offset: offset)
        }

        if !validateGetMappedRange(offset, rangeSize) {
            return SpanUInt8()
        }

        m_mappedRanges.add(WTFRangeSizeT(UInt(offset), UInt(offset + rangeSize)))
        m_mappedRanges.compact()

        if m_buffer.storageMode == .private || m_buffer.storageMode == .memoryless || m_buffer.length == 0 {
            return SpanUInt8()
        }

        return getBufferContents().subspan(offset, stdDynamicExtent)
    }
}
