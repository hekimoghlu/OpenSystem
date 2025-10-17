/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
#import "MBCBoardCommon.h"
#import "MBCMathUtilities.h"
#import "MBCShaderTypes.h"

#import <Metal/Metal.h>

/*!
 @abstract Defines the bottom / end V texture coordinate for the arrow tip section in arrow texture.
 Value comes from the vertical pixel posiition in texture where tip glow does not impact middle of arrow.
 */
#define MOVE_ARROW_TIP_TEXTURE_V_STOP   183.f / 256.f

/*!
 @abstract Defines the top / starting V texture coordinate for the arrow tail section in arrow texture.
 Value comes from the vertical position in texture before tail corner begins to round. 
 */
#define MOVE_ARROW_TAIL_TEXTURE_V_START  200.f / 256.f

/*!
 @abstract Minimum distance between two neighboring squares in horizontal or vertical direction.
 */
#define MIN_DISTANCE_BETWEEN_SQUARE_CENTERS 10.f

/*!
 @abstract Defines the length within the geometry of the tip and the tail portion for the move arrow.
 The tip and rounded part of tail will thus be same size no matter the total arrow length.
 */
#define MOVE_ARROW_WORLD_TIP_LENGTH     MOVE_ARROW_TIP_TEXTURE_V_STOP * MIN_DISTANCE_BETWEEN_SQUARE_CENTERS
#define MOVE_ARROW_WORLD_TAIL_LENGTH    (1.f - MOVE_ARROW_TAIL_TEXTURE_V_START) * MIN_DISTANCE_BETWEEN_SQUARE_CENTERS

/*!
 @abstract Manages the width of the arrow drawn for moves.  Used to scale the vertices in vertex shader
 and thus the total width of the arrow will be 2 times this value.
 */
#define MOVE_ARROW_HALF_WIDTH 5.f

/*!
 @abstract The arrow texture has an outer glow and thus the arrow tail and tip are inset away from the
 top and bottom of the texture. Use this to add more lenght to the arrow's geometry to account for this.
 */
#define MOVE_ARROW_LENGTH_ADJUSTMENT 2.f

/*!
 @abstract The alpha component value to use for last move arrow colors.
 */
#define MOVE_ARROW_LAST_MOVE_OPACITY 0.5f

#define MOVE_ARROW_VERTEX_COUNT 8

MBCPosition operator-(const MBCPosition & a, const MBCPosition & b) {
    MBCPosition    res;
    
    res[0]    = a[0]-b[0];
    res[1]    = a[1]-b[1];
    res[2]    = a[2]-b[2];

    return res;
}

@implementation MBCPieceInstance

- (float)pieceBaseRadius {
    switch (_type) {
        case KING:
            return kKingShadowScale;
        case QUEEN:
            return kQueenShadowScale;
        case PAWN:
            return kPawnShadowScale;
        case ROOK:
            return kRookShadowScale;
        case KNIGHT:
            return kKnightShadowScale;
        case BISHOP:
            return kBishopShadowScale;
        default:
            return 3.5f;
    }
}

@end

@implementation MBCBoardDecalInstance {
    vector_float3 _position;
}

- (instancetype)initWithPosition:(vector_float3)position {
    self = [super init];
    if (self) {
        _position = position;
        _rotate = NO;
        _uvScale = 1.f;
        _uvOrigin = simd_make_float2(0.f, 0.f);
        _quadVertexScale = 1.f;
        _visible = YES;
        _animateScale = NO;
        _color = simd_make_float3(1.f, 1.f, 1.f);
        
        [self updateModelMatrix];
    }
    return self;
}

- (void)updateModelMatrix {
    // Rotation
    // Quad is a plane in XY plane, rotate -90 degrees about X axis so lies in XZ plane.
    matrix_float4x4 matrix = matrix4x4_rotation(M_PI_2, kAxisRight);
    
    if (_rotate) {
        // Rotate 180 degrees about the y axis so text is not upside down based on viewing angle.
        matrix_float4x4 rotateY = matrix4x4_rotation(M_PI, kAxisUp);
        matrix = matrix_multiply(rotateY, matrix);
    }
    
    // Translation
    matrix_float4x4 translate = matrix4x4_translation(_position.x, _position.y, _position.z);
    
    _modelMatrix = simd_mul(translate, matrix);
}

- (void)setRotate:(BOOL)rotate {
    if (_rotate == rotate) {
        return;
    }
    _rotate = rotate;
    [self updateModelMatrix];
}

- (void)setPosition:(vector_float3)position {
    if (simd_equal(_position, position)) {
        return;
    }
    _position = position;
    [self updateModelMatrix];
}

@end

@implementation MBCArrowInstance {
    BOOL _isHint;
    MBCPiece _piece;
    MBCArrowVertex _arrowVertices[MOVE_ARROW_VERTEX_COUNT];
}

- (instancetype)initWithFromPosition:(MBCPosition)fromPosition
                          toPosition:(MBCPosition)toPosition
                               piece:(MBCPiece)piece
                              isHint:(BOOL)isHint {
    self = [super init];
    if (self) {
        _isHint = isHint;
        _piece = piece;
        
        BOOL isBlackPiece = Color(piece) == kBlackPiece;
        if (isHint) {
            _color = isBlackPiece ? simd_make_float4(0.f, 0.5f, 0.6f, 1.f) : simd_make_float4(0.75f, 0.5f, 0.f, 1.f);
        } else {
            _color = isBlackPiece ? simd_make_float4(0.f, 0.f, 0.f, MOVE_ARROW_LAST_MOVE_OPACITY) : simd_make_float4(1.f, 1.f, 1.f, MOVE_ARROW_LAST_MOVE_OPACITY);
        }
        
        [self initializeVertexPositions];
        [self computeModelMatrixFromPosition:fromPosition toPosition:toPosition];
    }
    return self;
}

- (void)initializeVertexPositions {
    // Initialize the vertices for the move arrow geometry. The positions are initially
    // in normalized [-1, 1] and UV coordinates are defined based on source texture.
    MBCArrowVertex kSimpleQuadVertices[] = {
        { { -1.0f,  -1.0f, }, { 0.0f, 0.0f} },
        { {  1.0f,  -1.0f, }, { 1.0f, 0.0f} },
        
        { { -1.0f,  -0.5f, }, { 0.0f, MOVE_ARROW_TIP_TEXTURE_V_STOP} },
        { {  1.0f,  -0.5f, }, { 1.0f, MOVE_ARROW_TIP_TEXTURE_V_STOP} },
        
        { { -1.0f,   0.5f, }, { 0.0f, MOVE_ARROW_TAIL_TEXTURE_V_START} },
        { {  1.0f,   0.5f, }, { 1.0f, MOVE_ARROW_TAIL_TEXTURE_V_START} },
        
        { { -1.0f,   1.0f, }, { 0.0f, 1.0f } },
        { {  1.0f,   1.0f, }, { 1.0f, 1.0f } }
    };
    
    for (int i = 0; i < MOVE_ARROW_VERTEX_COUNT; ++i) {
        kSimpleQuadVertices[i].position.x *= MOVE_ARROW_HALF_WIDTH;
    }
    
    memmove(_arrowVertices, kSimpleQuadVertices, sizeof(kSimpleQuadVertices));
}

- (void)computeModelMatrixFromPosition:(MBCPosition)fromPosition toPosition:(MBCPosition)toPosition {
    // Will use the total length of the arrow to calculate the vertex positions for
    // the geometry for the arrow. For the last move arrow, will offset the tip away
    // from the to square's center.
    const float tipOffset = _isHint ? 0.f : 2.f;
    const float halfTipOffset = tipOffset * 0.5f;
    _length = MOVE_ARROW_LENGTH_ADJUSTMENT + hypot(toPosition[0] - fromPosition[0], toPosition[2] - fromPosition[2]) - tipOffset;
    const float halfLength = _length * 0.5f;
    
    const float topQuadOuterY = halfLength;
    const float topQuadInnerY = topQuadOuterY - MOVE_ARROW_WORLD_TIP_LENGTH;
    const float bottomQuadOuterY = -halfLength;
    const float bottomQuadInnerY = -(halfLength - MOVE_ARROW_WORLD_TAIL_LENGTH);
    
    _arrowVertices[0].position.y = topQuadOuterY;
    _arrowVertices[1].position.y = topQuadOuterY;
    _arrowVertices[2].position.y = topQuadInnerY;
    _arrowVertices[3].position.y = topQuadInnerY;
    
    _arrowVertices[4].position.y = bottomQuadInnerY;
    _arrowVertices[5].position.y = bottomQuadInnerY;
    _arrowVertices[6].position.y = bottomQuadOuterY;
    _arrowVertices[7].position.y = bottomQuadOuterY;
    
    // Rotate - atan2(y,x) is the angle measured between the positive
    // x-axis and the ray from the origin to the point (x, y)
    float angle = atan2(toPosition[2] - fromPosition[2], toPosition[0] - fromPosition[0]);
    angle = (M_PI_2 - angle);
    
    matrix_float4x4 rotateY = matrix4x4_rotation(angle, kAxisUp);

    // Translate - find the midpoint, which will be position for the quad to render arrow.
    // centerY is different to alleviate z fighting of meshes overlap in y.
    const float centerX = (toPosition[0] + fromPosition[0]) * 0.5f - halfTipOffset * sin(angle);
    const float centerY = _isHint ? MBC_POSITION_Y_HINT_ARROW : MBC_POSITION_Y_LAST_MOVE_ARROW;
    const float centerZ = (toPosition[2] + fromPosition[2]) * 0.5f - halfTipOffset * cos(angle);
    matrix_float4x4 translateCenter = matrix4x4_translation(centerX, centerY, centerZ);
    
    // Create arrow model matrix by multiplying rotation, translation
    _modelMatrix = simd_mul(translateCenter, rotateY);
}

- (void)updateMTLVertexBuffer:(id<MTLBuffer>)buffer {
    MBCArrowVertex *vertices = (MBCArrowVertex *)[buffer contents];
    memcpy(vertices, _arrowVertices, sizeof(MBCArrowVertex) * 8);
}

- (BOOL)isBlackSideMove {
    return Color(_piece) == kBlackPiece;
}

@end
