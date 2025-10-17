/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#import "MBCMeshLoader.h"

// Include header shared between C code here, which executes Metal API commands, and .metal files
#import "MBCShaderTypes.h"

#import <ModelIO/ModelIO.h>
#import <simd/simd.h>

@implementation MBCSubmesh

- (nonnull instancetype) initWithModelIOSubmesh:(nonnull MDLSubmesh *)modelIOSubmesh
                                metalKitSubmesh:(nonnull MTKSubmesh *)metalKitSubmesh
                          metalKitTextureLoader:(nonnull MTKTextureLoader *)textureLoader {
    self = [super init];
    if (self) {
        _metalKitSubmmesh = metalKitSubmesh;
        
        // Note, textures are loaded outside of this method for submeshes since textures
        // may be swapped out in settings. They are therefore managed outside of the submesh
        // and do not need to be loaded using the data from the model file.
    }

    return self;
}

@end

@implementation MBCMesh {
    NSMutableArray<MBCSubmesh *> *_submeshes;
}

- (nonnull instancetype) initWithModelIOMesh:(nonnull MDLMesh *)modelIOMesh
                     modelIOVertexDescriptor:(nonnull MDLVertexDescriptor *)vertexDescriptor
                       metalKitTextureLoader:(nonnull MTKTextureLoader *)textureLoader
                                 metalDevice:(nonnull id<MTLDevice>)device
                                       error:(NSError * __nullable * __nullable)error {
    self = [super init];
    if (!self) {
        return nil;
    }

    // Have ModelIO create the tangents from mesh texture coordinates and normals
    [modelIOMesh addTangentBasisForTextureCoordinateAttributeNamed:MDLVertexAttributeTextureCoordinate
                                              normalAttributeNamed:MDLVertexAttributeNormal
                                             tangentAttributeNamed:MDLVertexAttributeTangent];

    // Have ModelIO create bitangents from mesh texture coordinates and the newly created tangents
    [modelIOMesh addTangentBasisForTextureCoordinateAttributeNamed:MDLVertexAttributeTextureCoordinate
                                             tangentAttributeNamed:MDLVertexAttributeTangent
                                           bitangentAttributeNamed:MDLVertexAttributeBitangent];

    // Apply the ModelIO vertex descriptor that the renderer created to match the Metal vertex descriptor.

    // Assigning a new vertex descriptor to a ModelIO mesh performs a re-layout of the vertex
    // vertex data.  In this case, rthe renderer created the ModelIO vertex descriptor so that the
    // layout of the vertices in the ModelIO mesh match the layout of vertices the Metal render
    // pipeline expects as input into its vertex shader

    // Note ModelIO must create tangents and bitangents (as done above) before this relayout occur
    // This is because Model IO's addTangentBasis methods only works with vertex data is all in
    // 32-bit floating-point.  The vertex descriptor applied, changes those floats into 16-bit
    // floats or other types from which ModelIO cannot produce tangents

    modelIOMesh.vertexDescriptor = vertexDescriptor;

    // Create the metalKit mesh which will contain the Metal buffer(s) with the mesh's vertex data
    //   and submeshes with info to draw the mesh
    MTKMesh* metalKitMesh = [[MTKMesh alloc] initWithMesh:modelIOMesh
                                                   device:device
                                                    error:error];

    _metalKitMesh = metalKitMesh;

    // There should always be the same number of MetalKit submeshes in the MetalKit mesh as there
    //   are Model IO submeshes in the Model IO mesh
    assert(metalKitMesh.submeshes.count == modelIOMesh.submeshes.count);

    // Create an array to hold this MBCMesh object's MBCSubmesh objects
    _submeshes = [[NSMutableArray alloc] initWithCapacity:metalKitMesh.submeshes.count];

    // Create an MBCSubmesh object for each submesh and a add it to the submesh's array
    for (NSUInteger index = 0; index < metalKitMesh.submeshes.count; index++) {
        // Create an app specific submesh to hold the MetalKit submesh
        MBCSubmesh *submesh = [[MBCSubmesh alloc] initWithModelIOSubmesh:modelIOMesh.submeshes[index]
                                                         metalKitSubmesh:metalKitMesh.submeshes[index]
                                                   metalKitTextureLoader:textureLoader];

        [_submeshes addObject:submesh];
    }

    return self;
}

+ (NSArray<MBCMesh*> *)newMeshesFromObject:(nonnull MDLObject*)object
                   modelIOVertexDescriptor:(nonnull MDLVertexDescriptor*)vertexDescriptor
                     metalKitTextureLoader:(nonnull MTKTextureLoader *)textureLoader
                               metalDevice:(nonnull id<MTLDevice>)device
                                     error:(NSError * __nullable * __nullable)error {

    NSMutableArray<MBCMesh *> *newMeshes = [[NSMutableArray alloc] init];

    // If this Model I/O  object is a mesh object (not a camera, light, or something else),
    // create an app-specific MBCMesh object from it
    if ([object isKindOfClass:[MDLMesh class]]) {
        MDLMesh* mesh = (MDLMesh*) object;

        MBCMesh *newMesh = [[MBCMesh alloc] initWithModelIOMesh:mesh
                                        modelIOVertexDescriptor:vertexDescriptor
                                          metalKitTextureLoader:textureLoader
                                                    metalDevice:device
                                                          error:error];

        [newMeshes addObject:newMesh];
    }

    // Recursively traverse the ModelIO asset hierarchy to find ModelIO meshes that are children
    // of this ModelIO object and create app-specific MBCMesh objects from those ModelIO meshes
    for (MDLObject *child in object.children) {
        NSArray<MBCMesh*> *childMeshes;

        childMeshes = [MBCMesh newMeshesFromObject:child
                           modelIOVertexDescriptor:vertexDescriptor
                             metalKitTextureLoader:textureLoader
                                       metalDevice:device
                                             error:error];

        [newMeshes addObjectsFromArray:childMeshes];
    }

    return newMeshes;
}

/// Uses Model I/O to load a model file at the given URL, create Model I/O vertex buffers, index buffers
///   and textures, applying the given Model I/O vertex descriptor to layout vertex attribute data
///   in the way that the Metal vertex shaders expect.
+ (nullable NSArray<MBCMesh *> *) newMeshesFromURL:(nonnull NSURL *)url
                           modelIOVertexDescriptor:(nonnull MDLVertexDescriptor *)vertexDescriptor
                                       metalDevice:(nonnull id<MTLDevice>)device
                                             error:(NSError * __nullable * __nullable)error {

    // Create a MetalKit mesh buffer allocator so that ModelIO will load mesh data directly into
    // Metal buffers accessible by the GPU
    MTKMeshBufferAllocator *bufferAllocator = [[MTKMeshBufferAllocator alloc] initWithDevice:device];

    // Use ModelIO to load the model file at the URL.  This returns a ModelIO asset object, which
    // contains a hierarchy of ModelIO objects composing a "scene" described by the model file.
    // This hierarchy may include lights, cameras, but, most importantly, mesh and submesh data
    // rendered with Metal
    MDLAsset *asset = [[MDLAsset alloc] initWithURL:url
                                   vertexDescriptor:nil
                                    bufferAllocator:bufferAllocator];

    NSAssert1(asset, @"Failed to open model file with given URL: %@", url.absoluteString);

    // Create a MetalKit texture loader to load material textures from files or the asset catalog
    //   into Metal textures
    MTKTextureLoader *textureLoader = [[MTKTextureLoader alloc] initWithDevice:device];

    NSMutableArray<MBCMesh *> *newMeshes = [[NSMutableArray alloc] init];

    // Traverse the ModelIO asset hierarchy to find ModelIO meshes and create app-specific
    // MBCMesh objects from those ModelIO meshes
    for (MDLObject* object in asset) {
        NSArray<MBCMesh *> *assetMeshes;

        assetMeshes = [MBCMesh newMeshesFromObject:object
                           modelIOVertexDescriptor:vertexDescriptor
                             metalKitTextureLoader:textureLoader
                                       metalDevice:device
                                             error:error];

        [newMeshes addObjectsFromArray:assetMeshes];
    }

    return newMeshes;
}

- (NSArray<MBCSubmesh *> *)submeshes {
    return _submeshes;
}

@end
