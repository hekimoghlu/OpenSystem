/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#import "MBCTuner.h"
#import "MBCBoardViewDraw.h"
#import "MBCBoardViewTextures.h"
#import "MBCController.h"
#import "MBCDrawStyle.h"

#import <simd/simd.h>

static MBCTuner *	sTuner;

@implementation MBCTunerStyle

- (void) updateFrom:(MBCDrawStyle *)drawStyle
{
	[fDiffuse setFloatValue:drawStyle->fDiffuse];
	[fSpecular setFloatValue:drawStyle->fSpecular];
	[fShininess setFloatValue:drawStyle->fShininess];
	[fAlpha setFloatValue:drawStyle->fAlpha];
}

- (void) updateTo:(MBCDrawStyle *)drawStyle
{
	drawStyle->fDiffuse		= [fDiffuse floatValue];
	drawStyle->fSpecular	= [fSpecular floatValue];
	drawStyle->fShininess	= [fShininess floatValue];
	drawStyle->fAlpha		= [fAlpha floatValue];
}

@end

@implementation MBCTuner 

+ (void) makeTuner
{
	//
	// Chess Tuner is intended to be run from inside the build directory
	// We create a styles link blindly
	//
	NSString * bndl	= [[NSBundle mainBundle] bundlePath]; 
	NSString * path = [bndl stringByDeletingLastPathComponent]; // .../build
	path			= [path stringByDeletingLastPathComponent]; // ...
	path			= [path stringByAppendingPathComponent:@"Styles"];
	bndl			= [bndl stringByAppendingPathComponent:
								@"Contents/Resources/Styles"];
	[[NSFileManager defaultManager] createSymbolicLinkAtPath:bndl
									pathContent:path];
										
	sTuner = [[MBCTuner alloc] init];
}

- (void) loadStyles
{
	fView	= [[MBCController controller] view];
	[fWhitePieceStyle updateFrom:[fView pieceDrawStyleAtIndex:0]];
    [fBlackPieceStyle updateFrom:[fView pieceDrawStyleAtIndex:1]];
    [fWhiteBoardStyle updateFrom:[fView boardDrawStyleAtIndex:0]];
	[fBlackBoardStyle updateFrom:[fView boardDrawStyleAtIndex:1]];
	[fBorderStyle updateFrom:[fView borderDrawStyle]];
	[fBoardReflectivity setFloatValue:fView.boardReflectivity];
	[fLabelIntensity setFloatValue:fView.labelIntensity];
    
    vector_float3 lightPosition = [fView lightPosition];
	[fLightPosX	setFloatValue:lightPosition.x];
	[fLightPosY	setFloatValue:lightPosition.y];
	[fLightPosZ	setFloatValue:lightPosition.z];
	[fAmbient setFloatValue:fView.ambient];
}

+ (void) loadStyles
{
	[sTuner loadStyles];
}

+ (void) saveStyles
{
}

- (id) init
{
	self = [super initWithWindowNibName:@"Tuner"];
	[[self window] orderFront:self];
	return self;
}

- (IBAction) updateWhitePieceStyle:(id)sender
{
    [fWhitePieceStyle updateTo:[fView pieceDrawStyleAtIndex:0]];
	[fView setNeedsDisplay:YES];
}

- (IBAction) updateBlackPieceStyle:(id)sender
{
    [fBlackPieceStyle updateTo:[fView pieceDrawStyleAtIndex:1]];
	[fView setNeedsDisplay:YES];
}

- (IBAction) updateWhiteBoardStyle:(id)sender
{
	[fWhiteBoardStyle updateTo:[fView boardDrawStyleAtIndex:0]];
	[fView setNeedsDisplay:YES];
}

- (IBAction) updateBlackBoardStyle:(id)sender
{
	[fBlackBoardStyle updateTo:[fView boardDrawStyleAtIndex:1]];
	[fView setNeedsDisplay:YES];
}

- (IBAction) updateBoardStyle:(id)sender
{
	[fBorderStyle updateTo:[fView borderDrawStyle]];
	fView.boardReflectivity	= [fBoardReflectivity floatValue];
	fView.labelIntensity = [fLabelIntensity floatValue];
	[fView setNeedsDisplay:YES];
}

- (IBAction) savePieceStyles:(id)sender
{
	[fView savePieceStyles];
}

- (IBAction) saveBoardStyles:(id)sender
{
	[fView saveBoardStyles];
}

static const char * sLightParams =
 "\tfloat   light_ambient		= %5.3ff\n"
 "\tGLfloat light_pos[4] 		= {%4.1ff, %4.1ff, %4.1ff, 1.0};\n";

- (IBAction) updateLight:(id)sender
{
    [fView setLightPosition:simd_make_float3([fLightPosX floatValue],
                                             [fLightPosY floatValue],
                                             [fLightPosZ floatValue])];
	fView.ambient = [fAmbient floatValue];
	[fView setNeedsDisplay:YES];
    
    vector_float3 lightPosition = [fView lightPosition];
    NSString *textString = [NSString stringWithFormat:[NSString stringWithUTF8String:sLightParams],
                            fView.ambient, lightPosition.x, lightPosition.y, lightPosition.z];
	[fLightParams setStringValue:textString];
}

@end

// Local Variables:
// mode:ObjC
// End:
