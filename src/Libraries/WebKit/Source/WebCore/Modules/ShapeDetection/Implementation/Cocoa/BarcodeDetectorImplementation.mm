/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#import "BarcodeDetectorImplementation.h"

#if HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)

#import "BarcodeDetectorOptionsInterface.h"
#import "BarcodeFormatInterface.h"
#import "DetectedBarcodeInterface.h"
#import "ImageBuffer.h"
#import "NativeImage.h"
#import "VisionUtilities.h"
#import <optional>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMallocInlines.h>
#import <pal/cocoa/VisionSoftLink.h>

namespace WebCore::ShapeDetection {

static BarcodeFormat convertSymbology(VNBarcodeSymbology symbology)
{
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyAztec()])
        return BarcodeFormat::Aztec;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyCodabar()])
        return BarcodeFormat::Codabar;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyCode39()])
        return BarcodeFormat::Code_39;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyCode39Checksum()])
        return BarcodeFormat::Code_39;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyCode39FullASCII()])
        return BarcodeFormat::Code_39;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyCode39FullASCIIChecksum()])
        return BarcodeFormat::Code_39;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyCode93()])
        return BarcodeFormat::Code_93;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyCode93i()])
        return BarcodeFormat::Code_93;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyCode128()])
        return BarcodeFormat::Code_128;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyDataMatrix()])
        return BarcodeFormat::Data_matrix;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyEAN8()])
        return BarcodeFormat::Ean_8;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyEAN13()])
        return BarcodeFormat::Ean_13;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyGS1DataBar()])
        return BarcodeFormat::Unknown;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyGS1DataBarExpanded()])
        return BarcodeFormat::Unknown;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyGS1DataBarLimited()])
        return BarcodeFormat::Unknown;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyI2of5()])
        return BarcodeFormat::Itf;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyI2of5Checksum()])
        return BarcodeFormat::Itf;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyITF14()])
        return BarcodeFormat::Itf;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyMicroPDF417()])
        return BarcodeFormat::Pdf417;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyMicroQR()])
        return BarcodeFormat::Qr_code;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyPDF417()])
        return BarcodeFormat::Pdf417;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyQR()])
        return BarcodeFormat::Qr_code;
    if ([symbology isEqual:PAL::get_Vision_VNBarcodeSymbologyUPCE()])
        return BarcodeFormat::Upc_e;
    return BarcodeFormat::Unknown;
}

static Vector<VNBarcodeSymbology> convertBarcodeFormat(BarcodeFormat barcodeFormat)
{
    switch (barcodeFormat) {
    case BarcodeFormat::Aztec:
        return { PAL::get_Vision_VNBarcodeSymbologyAztec() };
    case BarcodeFormat::Code_128:
        return { PAL::get_Vision_VNBarcodeSymbologyCode128() };
    case BarcodeFormat::Code_39:
        return { PAL::get_Vision_VNBarcodeSymbologyCode39(), PAL::get_Vision_VNBarcodeSymbologyCode39Checksum(), PAL::get_Vision_VNBarcodeSymbologyCode39FullASCII(), PAL::get_Vision_VNBarcodeSymbologyCode39FullASCIIChecksum() };
    case BarcodeFormat::Code_93:
        return { PAL::get_Vision_VNBarcodeSymbologyCode93(), PAL::get_Vision_VNBarcodeSymbologyCode93i() };
    case BarcodeFormat::Codabar:
        return { PAL::get_Vision_VNBarcodeSymbologyCodabar() };
    case BarcodeFormat::Data_matrix:
        return { PAL::get_Vision_VNBarcodeSymbologyDataMatrix() };
    case BarcodeFormat::Ean_13:
        return { PAL::get_Vision_VNBarcodeSymbologyEAN13() };
    case BarcodeFormat::Ean_8:
        return { PAL::get_Vision_VNBarcodeSymbologyEAN8() };
    case BarcodeFormat::Itf:
        return { PAL::get_Vision_VNBarcodeSymbologyI2of5(), PAL::get_Vision_VNBarcodeSymbologyI2of5Checksum(), PAL::get_Vision_VNBarcodeSymbologyITF14() };
    case BarcodeFormat::Pdf417:
        return { PAL::get_Vision_VNBarcodeSymbologyMicroPDF417(), PAL::get_Vision_VNBarcodeSymbologyPDF417() };
    case BarcodeFormat::Qr_code:
        return { PAL::get_Vision_VNBarcodeSymbologyMicroQR(), PAL::get_Vision_VNBarcodeSymbologyQR() };
    case BarcodeFormat::Unknown:
        return { PAL::get_Vision_VNBarcodeSymbologyGS1DataBar(), PAL::get_Vision_VNBarcodeSymbologyGS1DataBarExpanded(), PAL::get_Vision_VNBarcodeSymbologyGS1DataBarLimited() };
    case BarcodeFormat::Upc_a:
        return { };
    case BarcodeFormat::Upc_e:
        return { PAL::get_Vision_VNBarcodeSymbologyUPCE() };
    }
}

static std::optional<BarcodeDetectorImpl::BarcodeFormatSet> convertRequestedBarcodeFormatSet(const Vector<BarcodeFormat>& formats)
{
    // FIXME: "formats" is supposed to be optional

    BarcodeDetectorImpl::BarcodeFormatSet result;
    result.reserveInitialCapacity(formats.size());
    for (auto format : formats)
        result.add(format);
    return result;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(BarcodeDetectorImpl);

BarcodeDetectorImpl::BarcodeDetectorImpl(const BarcodeDetectorOptions& barcodeDetectorOptions)
    : m_requestedBarcodeFormatSet(convertRequestedBarcodeFormatSet(barcodeDetectorOptions.formats))
{
}

BarcodeDetectorImpl::~BarcodeDetectorImpl() = default;

static RetainPtr<VNDetectBarcodesRequest> request()
{
    // It's important that both getSupportedFormats() and detect() use a VNDetectBarcodesRequest that is
    // configured the same way. This function is intended to make sure both VNDetectBarcodesRequests are
    // configured accordingly.

    auto result = adoptNS([PAL::allocVNDetectBarcodesRequestInstance() init]);
    configureRequestToUseCPUOrGPU(result.get());
    return result;
}

void BarcodeDetectorImpl::getSupportedFormats(CompletionHandler<void(Vector<BarcodeFormat>&&)>&& completionHandler)
{
    NSError *error = nil;
    NSArray<VNBarcodeSymbology> *supportedSymbologies = [request() supportedSymbologiesAndReturnError:&error];

    BarcodeFormatSet barcodeFormatsSet;
    barcodeFormatsSet.reserveInitialCapacity(supportedSymbologies.count);
    for (VNBarcodeSymbology symbology in supportedSymbologies)
        barcodeFormatsSet.add(convertSymbology(symbology));

    auto barcodeFormatsVector = copyToVector(barcodeFormatsSet);
    std::sort(std::begin(barcodeFormatsVector), std::end(barcodeFormatsVector));

    completionHandler(WTFMove(barcodeFormatsVector));
}

void BarcodeDetectorImpl::detect(Ref<ImageBuffer>&& imageBuffer, CompletionHandler<void(Vector<DetectedBarcode>&&)>&& completionHandler)
{
    auto nativeImage = imageBuffer->copyNativeImage();
    if (!nativeImage) {
        completionHandler({ });
        return;
    }

    auto platformImage = nativeImage->platformImage();
    if (!platformImage) {
        completionHandler({ });
        return;
    }

    auto request = ShapeDetection::request();

    if (m_requestedBarcodeFormatSet) {
        NSMutableSet<VNBarcodeSymbology> *requestedSymbologies = [NSMutableSet setWithCapacity:m_requestedBarcodeFormatSet->size()];
        for (auto barcodeFormat : *m_requestedBarcodeFormatSet) {
            for (auto symbology : convertBarcodeFormat(barcodeFormat))
                [requestedSymbologies addObject:symbology];
        }
        request.get().symbologies = requestedSymbologies.allObjects;
    }

    auto imageRequestHandler = adoptNS([PAL::allocVNImageRequestHandlerInstance() initWithCGImage:platformImage.get() options:@{ }]);

    NSError *error = nil;
    auto result = [imageRequestHandler performRequests:@[request.get()] error:&error];
    if (!result || error) {
        completionHandler({ });
        return;
    }

    Vector<DetectedBarcode> results;
    results.reserveInitialCapacity(request.get().results.count);
    for (VNBarcodeObservation *observation in request.get().results) {
        results.append({
            convertRectFromVisionToWeb(nativeImage->size(), observation.boundingBox),
            observation.payloadStringValue,
            convertSymbology(observation.symbology),
            convertCornerPoints(nativeImage->size(), observation),
        });
    }

    completionHandler(WTFMove(results));
}

} // namespace WebCore::ShapeDetection

#endif // HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)
