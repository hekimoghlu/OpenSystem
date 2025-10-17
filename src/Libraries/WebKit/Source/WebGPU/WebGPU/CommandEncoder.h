/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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

#import "CommandBuffer.h"
#import "CommandsMixin.h"
#import "SwiftCXXThunk.h"
#import "WebGPU.h"
#import "WebGPUExt.h"
#import <wtf/FastMalloc.h>
#import <wtf/Function.h>
#import <wtf/Ref.h>
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/RetainReleaseSwift.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/Vector.h>
#import <wtf/WeakPtr.h>

@interface TextureAndClearColor : NSObject
- (instancetype)initWithTexture:(id<MTLTexture>)texture NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@property (nonatomic) id<MTLTexture> texture;
@property (nonatomic) MTLClearColor clearColor;
@property (nonatomic) NSUInteger depthPlane;
@end

struct WGPUCommandEncoderImpl {
};

namespace WebGPU {

class Buffer;
class CommandBuffer;
class ComputePassEncoder;
class Device;
class QuerySet;
class RenderPassEncoder;
class Sampler;
class Texture;

// https://gpuweb.github.io/gpuweb/#gpucommandencoder
class CommandEncoder : public CommandsMixin, public RefCountedAndCanMakeWeakPtr<CommandEncoder>, public WGPUCommandEncoderImpl {
    WTF_MAKE_TZONE_ALLOCATED(CommandEncoder);
public:
    static Ref<CommandEncoder> create(id<MTLCommandBuffer> commandBuffer, Device& device, uint64_t uniqueId)
    {
        return adoptRef(*new CommandEncoder(commandBuffer, device, uniqueId));
    }
    static Ref<CommandEncoder> createInvalid(Device& device)
    {
        return adoptRef(*new CommandEncoder(device));
    }

    ~CommandEncoder();

    Ref<ComputePassEncoder> beginComputePass(const WGPUComputePassDescriptor&) HAS_SWIFTCXX_THUNK;
    Ref<RenderPassEncoder> beginRenderPass(const WGPURenderPassDescriptor&) HAS_SWIFTCXX_THUNK;
    void copyBufferToBuffer(const Buffer& source, uint64_t sourceOffset, Buffer& destination, uint64_t destinationOffset, uint64_t size) HAS_SWIFTCXX_THUNK;
    void copyBufferToTexture(const WGPUImageCopyBuffer& source, const WGPUImageCopyTexture& destination, const WGPUExtent3D& copySize) HAS_SWIFTCXX_THUNK;
    void copyTextureToBuffer(const WGPUImageCopyTexture& source, const WGPUImageCopyBuffer& destination, const WGPUExtent3D& copySize) HAS_SWIFTCXX_THUNK;
    void copyTextureToTexture(const WGPUImageCopyTexture& source, const WGPUImageCopyTexture& destination, const WGPUExtent3D& copySize) HAS_SWIFTCXX_THUNK;
    void clearBuffer(Buffer&, uint64_t offset, uint64_t size);
    Ref<CommandBuffer> finish(const WGPUCommandBufferDescriptor&);
    void insertDebugMarker(String&& markerLabel);
    void popDebugGroup();
    void pushDebugGroup(String&& groupLabel);
    void resolveQuerySet(const QuerySet&, uint32_t firstQuery, uint32_t queryCount, Buffer& destination, uint64_t destinationOffset);
    void writeTimestamp(QuerySet&, uint32_t queryIndex);
    void setLabel(String&&);

    Device& device() const { return m_device; }
    Ref<Device> protectedDevice() const SWIFT_RETURNS_INDEPENDENT_VALUE { return m_device; }

    bool isValid() const { return m_commandBuffer; }
    void lock(bool);
    bool isLocked() const { return m_state == EncoderState::Locked; }

    bool isFinished() const { return m_state == EncoderState::Ended; }

    id<MTLBlitCommandEncoder> ensureBlitCommandEncoder();
    void finalizeBlitCommandEncoder();

    void runClearEncoder(NSMutableDictionary<NSNumber*, TextureAndClearColor*> *attachmentsToClear, id<MTLTexture> depthStencilAttachmentToClear, bool depthAttachmentToClear, bool stencilAttachmentToClear, float depthClearValue = 0, uint32_t stencilClearValue = 0, id<MTLRenderCommandEncoder> existingEncoder = nil);
    static void clearTextureIfNeeded(const WGPUImageCopyTexture&, NSUInteger, const Device&, id<MTLBlitCommandEncoder>);
    static void clearTextureIfNeeded(Texture&, NSUInteger, NSUInteger, const Device&, id<MTLBlitCommandEncoder>);
    void makeInvalid(NSString*);
    void makeSubmitInvalid(NSString* = nil);
    void incrementBufferMapCount();
    void decrementBufferMapCount();
    void endEncoding(id<MTLCommandEncoder>);
    void setLastError(NSString*);
    bool waitForCommandBufferCompletion();
    bool encoderIsCurrent(id<MTLCommandEncoder>) const;
    bool submitWillBeInvalid() const { return m_makeSubmitInvalid; }
    void addBuffer(id<MTLBuffer>);
    void addBuffer(id<MTLCounterSampleBuffer>);
    void addICB(id<MTLIndirectCommandBuffer>);
    void addTexture(id<MTLTexture>);
    void addTexture(const Texture&);
    void addSampler(const Sampler&);
    id<MTLCommandBuffer> commandBuffer() const;
    void setExistingEncoder(id<MTLCommandEncoder>);
    void generateInvalidEncoderStateError();
    bool validateClearBuffer(const Buffer&, uint64_t offset, uint64_t size);
    static void trackEncoder(CommandEncoder&, WeakHashSet<CommandEncoder>&);
    uint64_t uniqueId() const { return m_uniqueId; }
    NSMutableSet<id<MTLCounterSampleBuffer>> *timestampBuffers() const { return m_retainedTimestampBuffers; };
    void addOnCommitHandler(Function<bool(CommandBuffer&)>&&);

private:
    CommandEncoder(id<MTLCommandBuffer>, Device&, uint64_t uniqueId);
    CommandEncoder(Device&);

private PUBLIC_IN_WEBGPU_SWIFT:
    NSString* errorValidatingCopyBufferToBuffer(const Buffer& source, uint64_t sourceOffset, const Buffer& destination, uint64_t destinationOffset, uint64_t size);
private:
    NSString* validateFinishError() const;
    bool validatePopDebugGroup() const;
private PUBLIC_IN_WEBGPU_SWIFT:
    NSString* errorValidatingComputePassDescriptor(const WGPUComputePassDescriptor&) const;
    NSString* errorValidatingRenderPassDescriptor(const WGPURenderPassDescriptor&) const;
    void clearTextureIfNeeded(const WGPUImageCopyTexture&, NSUInteger);
private:
    NSString* errorValidatingImageCopyBuffer(const WGPUImageCopyBuffer&) const;
private PUBLIC_IN_WEBGPU_SWIFT:
    NSString* errorValidatingCopyBufferToTexture(const WGPUImageCopyBuffer&, const WGPUImageCopyTexture&, const WGPUExtent3D&) const;
    NSString* errorValidatingCopyTextureToBuffer(const WGPUImageCopyTexture&, const WGPUImageCopyBuffer&, const WGPUExtent3D&) const;
    NSString* errorValidatingCopyTextureToTexture(const WGPUImageCopyTexture& source, const WGPUImageCopyTexture& destination, const WGPUExtent3D& copySize) const;
private:
    void discardCommandBuffer();

    RefPtr<CommandBuffer> protectedCachedCommandBuffer() const { return m_cachedCommandBuffer.get(); }
    void retainTimestampsForOneUpdateLoop();

private PUBLIC_IN_WEBGPU_SWIFT:
    id<MTLCommandBuffer> m_commandBuffer { nil };
    id<MTLCommandEncoder> m_existingCommandEncoder { nil };
private:
    id<MTLSharedEvent> m_abortCommandBuffer { nil };
    id<MTLBlitCommandEncoder> m_blitCommandEncoder { nil };

    uint64_t m_debugGroupStackSize { 0 };
    ThreadSafeWeakPtr<CommandBuffer> m_cachedCommandBuffer;
    NSString* m_lastErrorString { nil };
    int m_bufferMapCount { 0 };
    bool m_makeSubmitInvalid { false };

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
    NSMutableSet<id<MTLTexture>> *m_managedTextures { nil };
    NSMutableSet<id<MTLBuffer>> *m_managedBuffers { nil };
#endif
    NSMutableSet<id<MTLIndirectCommandBuffer>> *m_retainedICBs { nil };
    NSMutableSet<id<MTLTexture>> *m_retainedTextures { nil };
    NSMutableSet<id<MTLBuffer>> *m_retainedBuffers { nil };
    HashSet<RefPtr<const Sampler>> m_retainedSamplers;
    NSMutableSet<id<MTLCounterSampleBuffer>> *m_retainedTimestampBuffers { nil };
    Vector<Function<bool(CommandBuffer&)>> m_onCommitHandlers;
    id<MTLSharedEvent> m_sharedEvent { nil };
    uint64_t m_sharedEventSignalValue { 0 };
private PUBLIC_IN_WEBGPU_SWIFT:
    const Ref<Device> m_device;
    uint64_t m_uniqueId;
private:
} SWIFT_SHARED_REFERENCE(refCommandEncoder, derefCommandEncoder);

} // namespace WebGPU

inline void refCommandEncoder(WebGPU::CommandEncoder* obj)
{
    WTF::ref(obj);
}

inline void derefCommandEncoder(WebGPU::CommandEncoder* obj)
{
    WTF::deref(obj);
}
