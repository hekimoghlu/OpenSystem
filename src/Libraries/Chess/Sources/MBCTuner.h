/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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
#import <Cocoa/Cocoa.h>
#import "MBCBoardViewInterface.h"

@class MBCDrawStyle;

@interface MBCTunerStyle : NSObject {
	IBOutlet id fDiffuse;
	IBOutlet id	fSpecular;
	IBOutlet id fShininess;
	IBOutlet id fAlpha;
};

- (void) updateFrom:(MBCDrawStyle *)drawStyle;
- (void) updateTo:(MBCDrawStyle *)drawStyle;

@end

@interface MBCTuner : NSWindowController
{
	NSView<MBCBoardViewInterface> * fView;
	IBOutlet MBCTunerStyle *	    fWhitePieceStyle;
	IBOutlet MBCTunerStyle * 	    fBlackPieceStyle;
	IBOutlet MBCTunerStyle * 	    fWhiteBoardStyle;
	IBOutlet MBCTunerStyle * 	    fBlackBoardStyle;
	IBOutlet MBCTunerStyle * 	    fBorderStyle;
	IBOutlet id					    fBoardReflectivity;
	IBOutlet id					    fLabelIntensity;
	IBOutlet id					    fLightPosX;
	IBOutlet id					    fLightPosY;
	IBOutlet id					    fLightPosZ;
	IBOutlet id					    fAmbient;
	IBOutlet id					    fLightParams;
}

+ (void) makeTuner;
+ (void) loadStyles;

- (IBAction) updateWhitePieceStyle:(id)sender;
- (IBAction) updateBlackPieceStyle:(id)sender;
- (IBAction) updateWhiteBoardStyle:(id)sender;
- (IBAction) updateBlackBoardStyle:(id)sender;
- (IBAction) updateBoardStyle:(id)sender;
- (IBAction) savePieceStyles:(id)sender;
- (IBAction) saveBoardStyles:(id)sender;
- (IBAction) updateLight:(id)sender;

@end

// Local Variables:
// mode:ObjC
// End:
