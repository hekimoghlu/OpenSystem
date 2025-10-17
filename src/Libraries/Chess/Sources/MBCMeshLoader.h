/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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
#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>
#import <Metal/Metal.h>

@class MDLVertexDescriptor;

/*!
 @abstract App specific submesh class containing data to draw a submesh
 */
@interface MBCSubmesh : NSObject

/*!
 @abstract A MetalKit submesh mesh containing the primitive type, index buffer, and index count used to draw all or part of its parent MBCMesh object
 */
@property (nonatomic, readonly, nonnull) MTKSubmesh *metalKitSubmmesh;

@end

/*!
 @abstract App specific mesh class containing vertex data describing the mesh and submesh object describing how to draw parts of the mesh
 */
@interface MBCMesh : NSObject

/*!
 @abstract newMeshesFromURL:modelIOVertexDescriptor:metalDevice:error:
 @param url  The location of a model file in a format supported by Model I/O, such as OBJ, ABC, or USD
 @param vertexDescriptor  Defines the layout ModelIO will use to arrange the vertex data
 @param device Metal device from renderer
 @param error Optional error that will be set if error occurs during loading
 @discussion Constructs an array of meshes from the provided file URL
 */
+ (nullable NSArray<MBCMesh *> *) newMeshesFromURL:(nonnull NSURL *)url
                            modelIOVertexDescriptor:(nonnull MDLVertexDescriptor *)vertexDescriptor
                                        metalDevice:(nonnull id<MTLDevice>)device
                                              error:(NSError * __nullable * __nullable)error;

/*!
 @abstract A MetalKit mesh containing vertex buffers describing the shape of the mesh
 */
@property (nonatomic, readonly, nonnull) MTKMesh *metalKitMesh;

/*!
 @abstract An array of MBCSubmesh objects containing buffers and data with which the render can draw
 including material data to set in a MTLRenderCommandEncoder object for draw the submesh
 */
@property (nonatomic, readonly, nonnull) NSArray<MBCSubmesh*> *submeshes;

@end
