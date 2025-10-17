/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
#import "config.h"
#import "SharedBuffer.h"

#import "WebCoreJITOperations.h"
#import "WebCoreObjCExtras.h"
#import <JavaScriptCore/InitializeThreading.h>
#import <string.h>
#import <wtf/MainThread.h>
#import <wtf/StdLibExtras.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/cocoa/VectorCocoa.h>

#import <pal/cf/CoreMediaSoftLink.h>

@interface WebCoreSharedBufferData : NSData
- (instancetype)initWithDataSegment:(const WebCore::DataSegment&)dataSegment position:(NSUInteger)position size:(NSUInteger)size;
@end

@implementation WebCoreSharedBufferData {
    RefPtr<const WebCore::DataSegment> _dataSegment;
    NSUInteger _position;
    NSUInteger _size;
}

+ (void)initialize
{
#if !USE(WEB_THREAD)
    JSC::initialize();
    WTF::initializeMainThread();
    WebCore::populateJITOperations();
#endif // !USE(WEB_THREAD)
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([WebCoreSharedBufferData class], self))
        return;

    [super dealloc];
}

- (instancetype)initWithDataSegment:(const WebCore::DataSegment&)dataSegment position:(NSUInteger)position size:(NSUInteger)size
{
    if (!(self = [super init]))
        return nil;

    RELEASE_ASSERT(position <= dataSegment.size());
    RELEASE_ASSERT(size <= dataSegment.size() - position);
    _dataSegment = &dataSegment;
    _position = position;
    _size = size;
    return self;
}

- (NSUInteger)length
{
    return _size;
}

- (const void *)bytes
{
    return _dataSegment->span().subspan(_position).data();
}

@end

namespace WebCore {

Ref<FragmentedSharedBuffer> FragmentedSharedBuffer::create(NSData *data)
{
    return adoptRef(*new FragmentedSharedBuffer(bridge_cast(data)));
}

void FragmentedSharedBuffer::append(NSData *data)
{
    ASSERT(!m_contiguous);
    return append(bridge_cast(data));
}

static void FreeDataSegment(void* refcon, void*, size_t)
{
    auto* buffer = reinterpret_cast<const DataSegment*>(refcon);
    buffer->deref();
}

RetainPtr<CMBlockBufferRef> FragmentedSharedBuffer::createCMBlockBuffer() const
{
    auto segmentToCMBlockBuffer = [] (const DataSegment& segment) -> RetainPtr<CMBlockBufferRef> {
        // From CMBlockBufferCustomBlockSource documentation:
        // Note that for 64-bit architectures, this struct contains misaligned function pointers.
        // To avoid link-time issues, it is recommended that clients fill CMBlockBufferCustomBlockSource's function pointer fields
        // by using assignment statements, rather than declaring them as global or static structs.
        CMBlockBufferCustomBlockSource allocator;
        allocator.version = 0;
        allocator.AllocateBlock = nullptr;
        allocator.FreeBlock = FreeDataSegment;
        allocator.refCon = const_cast<DataSegment*>(&segment);
        segment.ref();
        CMBlockBufferRef partialBuffer = nullptr;
        if (PAL::CMBlockBufferCreateWithMemoryBlock(nullptr, const_cast<uint8_t*>(segment.span().data()), segment.size(), nullptr, &allocator, 0, segment.size(), 0, &partialBuffer) != kCMBlockBufferNoErr)
            return nullptr;
        return adoptCF(partialBuffer);
    };

    if (hasOneSegment() && !isEmpty())
        return segmentToCMBlockBuffer(m_segments[0].segment);

    CMBlockBufferRef rawBlockBuffer = nullptr;
    auto err = PAL::CMBlockBufferCreateEmpty(kCFAllocatorDefault, isEmpty() ? 0 : m_segments.size(), 0, &rawBlockBuffer);
    if (err != kCMBlockBufferNoErr || !rawBlockBuffer)
        return nullptr;
    auto blockBuffer = adoptCF(rawBlockBuffer);

    if (isEmpty())
        return blockBuffer;

    for (auto& segment : m_segments) {
        if (!segment.segment->size())
            continue;
        auto partialBuffer = segmentToCMBlockBuffer(segment.segment);
        if (!partialBuffer)
            return nullptr;
        if (PAL::CMBlockBufferAppendBufferReference(rawBlockBuffer, partialBuffer.get(), 0, 0, 0) != kCMBlockBufferNoErr)
            return nullptr;
    }
    return blockBuffer;
}

RetainPtr<NSData> SharedBuffer::createNSData() const
{
    return bridge_cast(createCFData());
}

RetainPtr<CFDataRef> SharedBuffer::createCFData() const
{
    if (!m_segments.size())
        return adoptCF(CFDataCreate(nullptr, nullptr, 0));
    return bridge_cast(m_segments[0].segment->createNSData());
}

RetainPtr<NSArray> FragmentedSharedBuffer::createNSDataArray() const
{
    return createNSArray(m_segments, [] (auto& segment) {
        return segment.segment->createNSData();
    });
}

RetainPtr<NSData> DataSegment::createNSData() const
{
    return adoptNS([[WebCoreSharedBufferData alloc] initWithDataSegment:*this position:0 size:size()]);
}

void DataSegment::iterate(CFDataRef data, const Function<void(std::span<const uint8_t>)>& apply) const
{
    [(__bridge NSData *)data enumerateByteRangesUsingBlock:^(const void *bytes, NSRange byteRange, BOOL *) {
        apply(unsafeMakeSpan(static_cast<const uint8_t*>(bytes), byteRange.length));
    }];
}

RetainPtr<NSData> SharedBufferDataView::createNSData() const
{
    return adoptNS([[WebCoreSharedBufferData alloc] initWithDataSegment:m_segment.get() position:m_positionWithinSegment size:size()]);
}

} // namespace WebCore
