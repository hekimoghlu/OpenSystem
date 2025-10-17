/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#ifndef MBCShaderTypes_h
#define MBCShaderTypes_h

#include <simd/simd.h>

/*!
 @abstract Total number of lights in the scene (1 main,  rest are fill)
 */
#define MBC_TOTAL_LIGHT_COUNT 4
#define MBC_MAIN_LIGHT_INDEX 0
#define MBC_SPOT_LIGHT_INDEX 1
#define MBC_FILL_LIGHT_START_INDEX 2
#define MBC_FILL_LIGHT_COUNT MBC_TOTAL_LIGHT_COUNT - 2

#ifndef __METAL_VERSION__
/*!
 @typedef `packed_float3`
 @abstract 96-bit 3 component float vector type
 */
typedef struct __attribute__ ((packed)) packed_float3 {
    float x;
    float y;
    float z;
} packed_float3;

#else

/*!
 @abstract Common linear sampler used across multiple metal shdar files.
 */
constexpr metal::sampler linearSampler(metal::address::repeat,
                                metal::mip_filter::linear,
                                metal::mag_filter::linear,
                                metal::min_filter::linear);

#endif

/*!
 @typedef MBCBufferIndices
 @abstract Buffer index values shared between shader and C code to ensure Metal
 shader buffer inputs match Metal API buffer set calls
 */
typedef enum MBCBufferIndices {
    MBCBufferIndexMeshPositions     = 0,
    MBCBufferIndexMeshGenerics      = 1,
    MBCBufferIndexFrameData         = 2,
    MBCBufferIndexRenderableData    = 3,
    MBCBufferIndexMaterialData      = 4,
    MBCBufferIndexLightsData        = 5,
    MBCBufferIndexTriangleIndices   = 6
} MBCBufferIndices;

/*!
 @typedef MBCVertexAttributes
 @abstract Attribute index values shared between shader and C code to ensure Metal shader
 vertex attribute indices match the Metal API vertex descriptor attribute indices
 */
typedef enum MBCVertexAttributes {
    MBCVertexAttributePosition  = 0,
    MBCVertexAttributeTexcoord  = 1,
    MBCVertexAttributeNormal    = 2,
    MBCVertexAttributeTangent   = 3,
    MBCVertexAttributeBitangent = 4
} MBCVertexAttributes;

/*!
 @typedef MBCTextureIndices
 @abstract Texture index values shared between shader and C code to ensure Metal shader
 texture indices match indices of Metal API texture set calls
 */
typedef enum MBCTextureIndices {
	MBCTextureIndexBaseColor           = 0,
    MBCTextureIndexNormal              = 1,
    MBCTextureIndexMetallic            = 2,
    MBCTextureIndexRoughnessAO         = 3,
    MBCTextureIndexShadow              = 4,
    MBCTextureIndexReflection          = 5,
    MBCTextureIndexIrradianceMap       = 6,
} MBCTextureIndices;

/*!
 @typedef MBCFunctionConstantIndex
 @abstract Define the indices for function constants used in Metal shaders
 */
typedef enum MBCFunctionConstantIndex {
    MBCFunctionConstantIndexUseMaterialMaps            = 0,
    MBCFunctionConstantIndexSampleReflectionMap        = 1,
    MBCFunctionConstantIndexSampleIrradianceMap        = 2,
    MBCFunctionConstantIndexUseBoardLighting           = 3,
} MBCFunctionConstantIndex;

/*!
 @typedef MBCFrameData
 @abstract Structures shared between shader and C code to ensure the layout of common per frame data
 */
typedef struct {
    matrix_float4x4 projection_matrix;
    matrix_float4x4 view_matrix;
    matrix_float4x4 view_projection_matrix;
    matrix_float4x4 shadow_vp_matrix;
    matrix_float4x4 shadow_vp_texture_matrix;
    vector_float3 camera_position;
    vector_float3 light_positions[MBC_TOTAL_LIGHT_COUNT];
    float main_light_specular_intensity;
    vector_uint2 viewport_size;
    uint8_t light_count;
} MBCFrameData;

/*!
 @typedef MBCSimpleMaterial
 @abstract Stores simple material information when material maps not being used
 */
typedef struct {
    /*!
     @abstract Base color value for the material (albedo)
     */
    vector_float4 base_color;
    
    /*!
     @abstract Specular color for the material
     */
    vector_float3 specular_color;
    
    /*!
     @abstract The shininess of the surface (from 0 being smooth to 1 being rough)
     */
    float roughness; // Shininess of the surface, 0 smooth, 1 rough
    
    /*!
     @abstract 0 for dielectric material, 1 for metal material
     */
    float metallic;
    
    /*!
     @abstract Defines how much light reaches surface
     */
    float ambientOcclusion;
} MBCSimpleMaterial;

/*!
 @typedef MBCRenderableData
 @abstract Per instance data for each rendered object
 */
typedef struct {
    /*!
     @abstract Model matrix to transform vertices in vertex shader
     */
    matrix_float4x4 model_matrix;
    
    /*!
     @abstract Alpha value for rendering transparent instances
     */
    float alpha;
} MBCRenderableData;

/*!
 @typedef MBCArrowRenderableData
 @abstract Per instance data for sending hint or move arrow instance data to the GPU.
 */
typedef struct {
    /*!
     @abstract Model matrix to transform vertices in vertex shader
     */
    matrix_float4x4 model_matrix;
    
    /*!
     @abstract Color for the arrow
     */
    vector_float4 color;
    
    /*!
     @abstract Used to animate the color of the arrow
     */
    float animation_offset;
    
    /*!
     @abstract Total arrow length from tip to tail
     */
    float length;
} MBCArrowRenderableData;

/*!
 @typedef MBCDecalRenderableData
 @abstract Per instance data for sending decal instance data to the GPU. The decals are 2D
 planar models that are drawn above the surface of the board and not part of board mesh.
 */
typedef struct {
    /*!
     @abstract Model matrix to transform vertices in vertex shader
     */
    matrix_float4x4 model_matrix;
    
    /*!
     @abstract Color for multiplying texture's RGB sample value.
     */
    vector_float3 color;
    
    /*!
     @abstract Scales quad XY position of individual vertices to manage label size on screen.
     For example, vertex at [-1, 1] becomes vertex at [-scale.x, scale.y]
     */
    float quad_vertex_scale;
    
    /*!
     @abstract Need to offset uv to get the correct label value from the DigitGrid texture.
     */
    vector_float2 uv_origin;
    
    /*!
     @abstract A multiplier to convert UV from [0, 1] range to smaller subset of the DigitGrid texture.
     Used in combination with the `uv_origin` to get correct UV coordinates for label instance.
     */
    float uv_scale;
} MBCDecalRenderableData;

/*!
 @typedef MBCLightData
 @abstract Represents the data needed for rendering for an individual light in scene. Used
 for all light types: main directional light, spotlight, and fill lights.
 */
typedef struct {
    vector_float3 position;
    vector_float3 light_color;
    vector_float3 specular_color;
    vector_float3 attenuation;
    float light_intensity;
    vector_float3 normalized_direction;
    
    /*!
     @abstract Direction of a spot light cone (from light source to surface)
     */
    vector_float3 spot_cone_direction;
    
    /*!
     @abstract Outer cone, no light outside of this angle.
     Light does not impact surface outside this angle
     */
    float spot_umbra_angle;
    
    /*!
     @abstract Inner cone where light is full intensity
     */
    float spot_penumbra_angle;
    
    /*!
     @abstract Max reach of the spot light's lighting capability
     */
    float spot_max_distance_falloff;
    
    /*!
     @abstract If true, the light direction from a fragment is simply position. If false,
     light direction is light position minus fragment world position.
     */
    bool direction_is_position;
} MBCLightData;

/*!
 @typedef MBCSimpleVertex
 @abstract Simple vertex used to render simple quads
 */
typedef struct {
    vector_float2 position;
} MBCSimpleVertex;

/*!
 @typedef MBCArrowVertex
 @abstract Vertex struct to send vertex data to arrow vertex shader to render move arrow geometry.
 */
typedef struct {
    vector_float2 position;
    vector_float2 uv;
} MBCArrowVertex;

/*!
 @typedef MBCShadowVertex
 @abstract Simple vertex with just position data for shadow vertex shader
 */
typedef struct {
    packed_float3 position;
} MBCShadowVertex;

#endif /* MBCShaderTypes_h */

