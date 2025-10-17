/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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

#if HAVE(VISION)

#import <Vision/Vision.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, Vision)

SOFT_LINK_CLASS_FOR_HEADER(PAL, VNDetectBarcodesRequest)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VNDetectFaceLandmarksRequest)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VNImageRequestHandler)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VNRecognizeTextRequest)

SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyAztec, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyCodabar, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyCode39, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyCode39Checksum, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyCode39FullASCII, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyCode39FullASCIIChecksum, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyCode93, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyCode93i, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyCode128, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyDataMatrix, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyEAN8, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyEAN13, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyGS1DataBar, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyGS1DataBarExpanded, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyGS1DataBarLimited, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyI2of5, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyI2of5Checksum, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyITF14, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyMicroPDF417, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyMicroQR, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyPDF417, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyQR, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Vision, VNBarcodeSymbologyUPCE, NSString *);

#endif // HAVE(VISION)
