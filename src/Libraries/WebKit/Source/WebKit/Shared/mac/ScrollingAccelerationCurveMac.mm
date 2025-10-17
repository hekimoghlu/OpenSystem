/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#import "ScrollingAccelerationCurve.h"

#if ENABLE(MOMENTUM_EVENT_DISPATCHER) && PLATFORM(MAC)

#import "Logging.h"
#import "NativeWebWheelEvent.h"
#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <pal/spi/cocoa/IOKitSPI.h>
#import <wtf/cf/TypeCastsCF.h>

namespace WebKit {

static float fromFixedPoint(float value)
{
    return value / 65536.0f;
}

static float fromCFNumber(CFNumberRef number)
{
    float value;
    CFNumberGetValue(number, kCFNumberFloatType, &value);
    return value;
}

static float readFixedPointParameter(NSDictionary *parameters, const char *key)
{
    return fromFixedPoint([[parameters objectForKey:@(key)] floatValue]);
}

static ScrollingAccelerationCurve fromIOHIDCurve(NSDictionary *parameters, float resolution, float frameRate)
{
    auto gainLinear = readFixedPointParameter(parameters, kHIDAccelGainLinearKey);
    auto gainParabolic = readFixedPointParameter(parameters, kHIDAccelGainParabolicKey);
    auto gainCubic = readFixedPointParameter(parameters, kHIDAccelGainCubicKey);
    auto gainQuartic = readFixedPointParameter(parameters, kHIDAccelGainQuarticKey);

    auto tangentSpeedLinear = readFixedPointParameter(parameters, kHIDAccelTangentSpeedLinearKey);
    auto tangentSpeedParabolicRoot = readFixedPointParameter(parameters, kHIDAccelTangentSpeedParabolicRootKey);

    return { gainLinear, gainParabolic, gainCubic, gainQuartic, tangentSpeedLinear, tangentSpeedParabolicRoot, resolution, frameRate };
}

static ScrollingAccelerationCurve fromIOHIDCurveArrayWithAcceleration(NSArray<NSDictionary *> *ioHIDCurves, float desiredAcceleration, float resolution, float frameRate)
{
    __block size_t currentIndex = 0;
    __block Vector<std::pair<float, ScrollingAccelerationCurve>> curves;

    [ioHIDCurves enumerateObjectsUsingBlock:^(NSDictionary *parameters, NSUInteger i, BOOL *) {
        auto curveAcceleration = readFixedPointParameter(parameters, kHIDAccelIndexKey);
        auto curve = fromIOHIDCurve(parameters, resolution, frameRate);

        if (desiredAcceleration > curveAcceleration)
            currentIndex = i;

        curves.append({ curveAcceleration, curve });
    }];

    // Interpolation if desiredAcceleration is in between two curves.
    if (curves[currentIndex].first < desiredAcceleration && (currentIndex + 1) < curves.size()) {
        const auto& lowCurve = curves[currentIndex];
        const auto& highCurve = curves[currentIndex + 1];
        float ratio = (desiredAcceleration - lowCurve.first) / (highCurve.first - lowCurve.first);
        return ScrollingAccelerationCurve::interpolate(lowCurve.second, highCurve.second, ratio);
    }

    return curves[currentIndex].second;
}

static RetainPtr<IOHIDEventSystemClientRef> createHIDClient()
{
    auto client = adoptCF(IOHIDEventSystemClientCreateWithType(nil, kIOHIDEventSystemClientTypePassive, nil));
    IOHIDEventSystemClientSetDispatchQueue(client.get(), dispatch_get_main_queue());
    IOHIDEventSystemClientActivate(client.get());
    return client;
}

static std::optional<ScrollingAccelerationCurve> fromIOHIDDevice(IOHIDEventSenderID senderID)
{
    static NeverDestroyed<RetainPtr<IOHIDEventSystemClientRef>> client;
    if (!client.get())
        client.get() = createHIDClient();

    RetainPtr<IOHIDServiceClientRef> ioHIDService = adoptCF(IOHIDEventSystemClientCopyServiceForRegistryID(client.get().get(), senderID));
    if (!ioHIDService) {
        RELEASE_LOG(ScrollAnimations, "ScrollingAccelerationCurve::fromIOHIDDevice did not find matching HID service");
        return std::nullopt;
    }

    auto curves = adoptCF(dynamic_cf_cast<CFArrayRef>(IOHIDServiceClientCopyProperty(ioHIDService.get(), CFSTR(kHIDScrollAccelParametricCurvesKey))));
    if (!curves) {
        RELEASE_LOG(ScrollAnimations, "ScrollingAccelerationCurve::fromIOHIDDevice failed to look up curves");
        return std::nullopt;
    }

    auto readFixedPointServiceKey = [&] (CFStringRef key) -> std::optional<float> {
        auto valueCF = adoptCF(dynamic_cf_cast<CFNumberRef>(IOHIDServiceClientCopyProperty(ioHIDService.get(), key)));
        if (!valueCF)
            return std::nullopt;
        return fromFixedPoint([(NSNumber *)valueCF.get() floatValue]);
    };

    auto scrollAcceleration = [&] () -> std::optional<float> {
        if (auto scrollAccelerationType = adoptCF(dynamic_cf_cast<CFStringRef>(IOHIDServiceClientCopyProperty(ioHIDService.get(), CFSTR("HIDScrollAccelerationType"))))) {
            if (auto acceleration = readFixedPointServiceKey(scrollAccelerationType.get()))
                return acceleration;
        }

        if (auto acceleration = readFixedPointServiceKey(CFSTR(kIOHIDMouseScrollAccelerationKey)))
            return acceleration;

        if (auto acceleration = readFixedPointServiceKey(CFSTR(kIOHIDScrollAccelerationKey)))
            return acceleration;

        return std::nullopt;
    }();
    if (!scrollAcceleration) {
        RELEASE_LOG(ScrollAnimations, "ScrollingAccelerationCurve::fromIOHIDDevice failed to look up acceleration");
        return std::nullopt;
    }

    auto resolution = readFixedPointServiceKey(CFSTR(kIOHIDScrollResolutionKey));
    if (!resolution) {
        RELEASE_LOG(ScrollAnimations, "ScrollingAccelerationCurve::fromIOHIDDevice failed to look up resolution");
        return std::nullopt;
    }

    static CFStringRef dispatchFrameRateKey = CFSTR("ScrollMomentumDispatchRate");
    static constexpr float defaultDispatchFrameRate = 60;
    auto frameRateCF = adoptCF(dynamic_cf_cast<CFNumberRef>(IOHIDServiceClientCopyProperty(ioHIDService.get(), dispatchFrameRateKey)));
    float frameRate = frameRateCF ? fromCFNumber(frameRateCF.get()) : defaultDispatchFrameRate;

    return fromIOHIDCurveArrayWithAcceleration((NSArray *)curves.get(), *scrollAcceleration, *resolution, frameRate);
}

std::optional<ScrollingAccelerationCurve> ScrollingAccelerationCurve::fromNativeWheelEvent(const NativeWebWheelEvent& nativeWebWheelEvent)
{
    NSEvent *event = nativeWebWheelEvent.nativeEvent();

    auto cgEvent = event.CGEvent;
    if (!cgEvent) {
        RELEASE_LOG(ScrollAnimations, "ScrollingAccelerationCurve::fromNativeWheelEvent did not find CG event");
        return std::nullopt;
    }

    auto hidEvent = adoptCF(CGEventCopyIOHIDEvent(cgEvent));
    if (!hidEvent) {
        RELEASE_LOG(ScrollAnimations, "ScrollingAccelerationCurve::fromNativeWheelEvent did not find HID event");
        return std::nullopt;
    }

    return fromIOHIDDevice(IOHIDEventGetSenderID(hidEvent.get()));
}

} // namespace WebKit

#endif // ENABLE(MOMENTUM_EVENT_DISPATCHER) && PLATFORM(MAC)
