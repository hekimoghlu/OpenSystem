/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#ifndef MBCBoardCommon_h
#define MBCBoardCommon_h

#import "MBCBoardEnums.h"

#import <Foundation/Foundation.h>
#import <math.h>
#import <simd/simd.h>

@protocol MTLBuffer;

const float kInHandPieceX       = 51.0f;
const float kInHandPieceXMTL    = 55.0f;
const float kInHandPieceZOffset = 3.0f;
const float kInHandPieceSize    = 8.0f;
const float kPromotionPieceX    = 50.0f;
const float kPromotionPieceZ    = 35.0f;
const float kBoardRadius        = 40.0f;
const float kBorderWidth        = 6.25f;
const float kBoardHeight        = 4.0f;
const float kBorderWidthMTL     = 10.0f;
const float kMinElevation       = 10.0f;
const float kMaxElevation       = 80.0f;

#define MBC_POSITION_Y_DELTA            0.05f
#define MBC_POSITION_Y_LABELS           0.1f
#define MBC_POSITION_Y_LAST_MOVE_ARROW  MBC_POSITION_Y_LABELS + MBC_POSITION_Y_DELTA
#define MBC_POSITION_Y_HINT_ARROW       MBC_POSITION_Y_LAST_MOVE_ARROW + MBC_POSITION_Y_DELTA
#define MBC_POSITION_Y_PIECE_SELECTION  MBC_POSITION_Y_HINT_ARROW + MBC_POSITION_Y_DELTA * 2.f

/*!
 @abstract vector representing the positive Y (Up) axis
 */
const vector_float3 kAxisUp = simd_make_float3(0.f, 1.f, 0.f);

/*!
 @abstract vector representing the positive X (Right) axis
 */
const vector_float3 kAxisRight = simd_make_float3(1.f, 0.f, 0.f);

/*!
 @abstract Defines the amount to scale the vertex positions for the pieceSelection quad. Values are based on the
 radius of each piece determined from the corresponding pieces 3d model.
 */
const float kSelectionScaleAmount   = 2.f;
const float kBishopSelectionScale   = 3.3162 + kSelectionScaleAmount;
const float kKingSelectionScale     = 3.8792 + kSelectionScaleAmount;
const float kKnightSelectionScale   = 3.6500 + kSelectionScaleAmount;
const float kPawnSelectionScale     = 3.0872 + kSelectionScaleAmount;
const float kQueenSelectionScale    = 4.1000 + kSelectionScaleAmount;
const float kRookSelectionScale     = 3.1726 + kSelectionScaleAmount;
const float kPieceSelectionScales[] = {0.f, kKingSelectionScale, kQueenSelectionScale, kBishopSelectionScale, kKnightSelectionScale, kRookSelectionScale, kPawnSelectionScale};

/*!
 @abstract Defines the amount to scale the shadow caster disk for each piece. Values are based on the
 radius of each piece determined from the corresponding pieces 3d model.
 */
const float kPieceShadowScale   = 0.4f;
const float kBishopShadowScale  = 3.3162 + kPieceShadowScale;
const float kKingShadowScale    = 3.8792 + kPieceShadowScale;
const float kKnightShadowScale  = 3.6500 + kPieceShadowScale;
const float kPawnShadowScale    = 3.0872 + kPieceShadowScale;
const float kQueenShadowScale   = 4.1000 + kPieceShadowScale;
const float kRookShadowScale    = 3.1726 + kPieceShadowScale;

/*!
 @abstract MBCPosition
 @discussion Defines a structure to represent a position in 3D space with XYZ floating point components.
*/
struct MBCPosition {
    float     pos[3];

    operator float * () { return pos; }
    float & operator[](int i) { return pos[i]; }
    float   operator[](int i) const { return pos[i]; }
    float FlatDistance(const MBCPosition & other) {
        return hypot(pos[0]-other[0], pos[2]-other[2]);
    }
};

/*!
 @abstract operator-
 @param a Left hand value of subtraction operation (minuend)
 @param b Right hand value that is subtracted from a (subtrahend)
 @return A new MBCPosition computed as (ax - bx, ay - by, az - bz)
 @discussion Defines the subtraction operator for two MBCPosition references
*/
MBCPosition operator-(const MBCPosition & a, const MBCPosition & b);


#pragma mark - MBCPieceInstance

/*!
 @abstract MBCPieceInstance
 @discussion Encapsulates data about pieces on the board that are used for Metal rendering.
*/
@interface MBCPieceInstance : NSObject

/*!
 @abstract The type of chess piece (King, Queen, Pawn, etc)
*/
@property (nonatomic) MBCPieceCode type;

/*!
 @abstract One of kWhitePiece or kBlackPiece
*/
@property (nonatomic) MBCPieceCode color;

/*!
 @abstract The world position of the piece on fhe board
*/
@property (nonatomic) vector_float4 position;

/*!
 @abstract The current scale of chess piece
*/
@property (nonatomic) float scale;

/*!
 @abstract The current alpha for chess piece.
*/
@property (nonatomic) float alpha;

/*!
 @abstract The radius for the bottom of the piece for shadows
*/
@property (nonatomic, readonly) float pieceBaseRadius;

@end

#pragma mark - MBCBoardDecalInstance

/*!
 @abstract MBCBoardDecalInstance
 @discussion Encapsulates data about decal images on surface of board. For example, the crazy house
 game variant renders captured piece counts on side of board. Also used for selected piece graphic.
*/
@interface MBCBoardDecalInstance : NSObject

/*!
 @abstract Model matrix for positioning and rotating the vertices.
*/
@property (nonatomic, readonly) matrix_float4x4 modelMatrix;

/*!
 @abstract Called to get/set the position of the instance. Setter will update model matrix if needed.
 */
@property (nonatomic) vector_float3 position;

/*!
 @abstract If true, then will rotate the object 180 degrees around local Y axis.
 */
@property (nonatomic) BOOL rotate;

/*!
 @abstract Scale the vertex quad to desired world size since the quad has vertices with XY values of 0 or 1.
 The quad is a square, thus this value will be half the total width and height of quad in world space. Default is 1.
 */
@property (nonatomic) float quadVertexScale;

/*!
 @abstract The origin for the uv coordinate for this instance. The label digit texture is a 4x4 grid of values,
 thus, this is an offset to the correct region for label value. Default is [0, 0].
*/
@property (nonatomic) vector_float2 uvOrigin;

/*!
 @abstract If uvOrigin is not 0, 0, then uv region for this instance is a subregion of a texture. For example,
 the label digit grid is a 4x4 grid and thus the uvScale will be 0.25. Default is 1.0.
 */
@property (nonatomic) float uvScale;

/*!
 @abstract isVisible
 @discussion Whether or not to draw this instance on the current frame. Default is YES.
 */
@property (nonatomic, getter=isVisible) BOOL visible;

/*!
 @abstract color
 @discussion The color that will be used to multiply RGB value of color sampled from texture.
 */
@property (nonatomic) vector_float3 color;

/*!
 @abstract animateScale
 @discussion Whether or not to animate the geometry scale. Default is NO.
 */
@property (nonatomic) BOOL animateScale;

/*!
 @abstract initWithPosition:
 @param position The world position  for center of quad.
 @discussion Called to create a new object for drawing decal graphic on board
 */
- (instancetype)initWithPosition:(vector_float3)position;

@end


#pragma mark - MBCArrowInstance

@interface MBCArrowInstance : NSObject

/*!
 @abstract Model matrix for positioning and rotating the arrows.
*/
@property (nonatomic, readonly) matrix_float4x4 modelMatrix;

/*!
 @abstract The color of the arrow for rendering. Texture RGB is white, shader will use this
 value to calculate the final color of the arrow by multiplying RGB components.
 */
@property (nonatomic, readonly) vector_float4 color;

/*!
 @abstract The total length of the arrow geometry from tail to tip between centers of two squares
 */
@property (nonatomic, readonly) float length;

/*!
 @abstract Returns YES if this move belongs to the black piece side.
 */
@property (nonatomic, readonly) BOOL isBlackSideMove;

/*!
 @abstract initWithFromPosition:toPosition:isHint:pieceColor
 @param fromPosition The source square position for the tail of the arrow
 @param toPosition The destination square position for the tip of the arrow
 @param isHint YES - is a hint move arrow, NO - is last move arrow
 @discussion Creates a new instance of an arrow to store data to pass from the game to the renderer.
 */
- (instancetype)initWithFromPosition:(MBCPosition)fromPosition
                          toPosition:(MBCPosition)toPosition
                               piece:(MBCPiece)piece
                              isHint:(BOOL)isHint;

/*!
 @abstract updateMTLVertexBuffer:
 @param buffer MTLBuffer that stores all the MBCArrowVertex instances for rendering a move arrow
 @discussion Will copy the vertex position and uv coordinate data to the contents of the buffer.
 */
- (void)updateMTLVertexBuffer:(id<MTLBuffer>)buffer;

@end

#endif /* MBCBoardCommon_h */
