/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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

private let largeBufferSize = 32 * 1024 * 1024

public func writeBuffer(
    queue: WebGPU.Queue, buffer: WebGPU.Buffer, bufferOffset: UInt64, data: SpanUInt8
) {
    queue.writeBuffer(buffer, bufferOffset: bufferOffset, data: data)
}

extension WebGPU.Queue {
    public func writeBuffer(_ buffer: WebGPU.Buffer, bufferOffset: UInt64, data: SpanUInt8) {
        let device = self.device()
        guard let blitCommandEncoder = ensureBlitCommandEncoder() else {
            return
        }
        let noCopy = data.size() >= largeBufferSize
        guard device.device() != nil else {
            return
        }
        guard
            let tempBuffer =
                noCopy
                ? device.newBufferWithBytesNoCopy(
                    data.__dataUnsafe(), data.size(), MTLResourceOptions.storageModeShared)
                : device.newBufferWithBytes(
                    data.__dataUnsafe(), data.size(), MTLResourceOptions.storageModeShared)
        else {
            return
        }
        blitCommandEncoder.copy(
            from: tempBuffer, sourceOffset: 0, to: buffer.buffer(),
            destinationOffset: Int(bufferOffset),
            size: data.size())
        if noCopy {
            finalizeBlitCommandEncoder()
        }
    }
}
