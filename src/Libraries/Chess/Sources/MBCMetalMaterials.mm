/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
#import "MBCMetalMaterials.h"
#import "MBCDrawStyle.h"
#import "MBCMetalRenderer.h"

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

typedef NS_ENUM(NSInteger, MBCStyleColorIndex) {
    MBCStyleColorIndexWhite = 0,
    MBCStyleColorIndexBlack = 1
};

/*!
 @abstract Used to store the PBR material parameter textures for each type of Chess piece.
 */
@interface MBCPiecePBRTextures : NSObject

/*!
 @abstract The MTLTexture with normal map data for chess piece
 */
@property (nonatomic, strong) id<MTLTexture> normalMap;

/*!
 @abstract The MTLTexture with roughness and ambient occlustion  map data for chess piece
 */
@property (nonatomic, strong) id<MTLTexture> roughnessAmbientOcclusion;

@end

@implementation MBCPiecePBRTextures

@end

@interface MBCMetalMaterials () {
    /*!
     @abstract Material attributes loaded from Board.plist for current material (ie Styles/Grass/Board.plist)
    */
    NSDictionary *_boardAttributes;
    
    /*!
     @abstract Material attributes loaded from Piece.plist for current material (ie Styles/Marble/Piece.plist)
    */
    NSDictionary *_pieceAttributes;
    
    /*!
     @abstract Currently used board styles for board to define textures and material PBR properties
    */
    NSMutableDictionary<NSString *, MBCDrawStyle *> *_boardDrawStyles;
    
    /*!
     @abstract Currently used piece styles for white and black pieces
    */
    NSMutableDictionary<NSString *, NSArray<MBCDrawStyle *> *> *_whitePieceDrawStyles;
    NSMutableDictionary<NSString *, NSArray<MBCDrawStyle *> *> *_blackPieceDrawStyles;
    
    /*!
     @abstract Stores the data structures that store PBR textures for each type of piece. Same
     textures are used for both the white and black variant of same type of chess piece.
     */
    NSArray<MBCPiecePBRTextures *> *_piecePBRTextures;
    
    /*!
     @abstract Draw style used for selected promotion piece
    */
    MBCDrawStyle *_selectedPieceDrawStyle;
    
    /*!
     @abstract String for the current material style for pieces and board (@"Wood", @"Metal", @"Marble")
    */
    NSString *_styleName;
    
    /*!
     @abstract Common anisotropy value, may not be needed for Metal.
    */
    float _anisotropy;
    
    /*!
     @abstract Controls amount of board reflectivity when enabled
    */
    float _boardReflectivity;
    
    /*!
     @abstract MTKTextureLoader used to load MTLTextures for styles
    */
    MTKTextureLoader *_textureLoader;
    
    NSDictionary<NSString *, NSMutableSet<NSString *> *> *_styleUsageMap;
}

@end

@implementation MBCMetalMaterials

+ (instancetype)shared {
    static MBCMetalMaterials *sSharedInstance;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sSharedInstance = [[MBCMetalMaterials alloc] init];
    });
    
    return sSharedInstance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _anisotropy = 4.f;
        _boardReflectivity = 0.3f;
        
        _styleUsageMap = @{
            @"Wood" : [NSMutableSet set],
            @"Marble" : [NSMutableSet set],
            @"Metal" : [NSMutableSet set]
        };
        
        _boardDrawStyles = [[NSMutableDictionary alloc] initWithCapacity:3];
        _whitePieceDrawStyles = [[NSMutableDictionary alloc] initWithCapacity:3];
        _blackPieceDrawStyles = [[NSMutableDictionary alloc] initWithCapacity:3];
        
        _piecePBRTextures = nil;

        _selectedPieceDrawStyle = [[MBCDrawStyle alloc] init];
        
        _textureLoader = [[MTKTextureLoader alloc] initWithDevice:MBCMetalRenderer.defaultMTLDevice];
        
        [self loadCommonTextures];
    }
    return self;
}

- (void)initializePieceStyleArraysForStyle:(NSString *)style {
    NSMutableArray<MBCDrawStyle *> *blackStyles = [NSMutableArray arrayWithCapacity:7];
    NSMutableArray<MBCDrawStyle *> *whiteStyles = [NSMutableArray arrayWithCapacity:7];
    for (int i = 0; i < 7; ++i) {
        [blackStyles addObject:[[MBCDrawStyle alloc] init]];
        [whiteStyles addObject:[[MBCDrawStyle alloc] init]];
    }
    _whitePieceDrawStyles[style] = [[NSArray alloc] initWithArray:whiteStyles];
    _blackPieceDrawStyles[style] = [[NSArray alloc] initWithArray:blackStyles];
}

- (void)loadCommonTextures {
    _groundTexture = [self loadTextureWithFileName:@"BoardShadow.jpg" label:@"GroundPlane" isSRGB:YES generateMipmaps:YES];
    
    _edgeNotationTexture = [self loadTextureWithFileName:@"DigitGrid.png" label:@"DigitGrid" isSRGB:YES generateMipmaps:YES];
    _edgeNotationNormalTexture = [self loadTextureWithFileName:@"DigitGridNormals.jpg" label:@"DigitGridNormals" isSRGB:NO generateMipmaps:YES];
    
    _pieceSelectionTexture = [self loadTextureWithFileName:@"PieceSelection.png" label:@"PieceSelection" isSRGB:YES generateMipmaps:YES];
    
    _hintArrowTexture = [self loadTextureWithFileName:@"hintMoveArrow.png" label:@"HintMoveArrow" isSRGB:YES generateMipmaps:YES];
    _lastMoveArrowTexture = [self loadTextureWithFileName:@"lastMoveArrow.png" label:@"LastMoveArrow" isSRGB:YES generateMipmaps:YES];
    
    // Irradiance map (a cube map) needs to be loaded from mdl texture.
    NSDictionary *textureLoaderOptions = @{
        MTKTextureLoaderOptionSRGB : @(NO),
        MTKTextureLoaderOptionOrigin : MTKTextureLoaderOriginTopLeft,
        MTKTextureLoaderOptionGenerateMipmaps : @(NO),
        MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModePrivate)
    };
    
    MDLTexture *mdlTexture = [MDLTexture textureCubeWithImagesNamed:@[@"irradiance.png"]];
    _irradianceMap = [_textureLoader newTextureWithMDLTexture:mdlTexture
                                                      options:textureLoaderOptions
                                                        error:nil];
}

- (void)loadPiecePBRTextures {
    // Order is based upon the MBCPieceCode enum
    NSArray<NSString *> * pieceNamePrefixes = @[@"king", @"queen", @"bishop", @"knight", @"rook", @"pawn"];
    NSMutableArray<MBCPiecePBRTextures *> *piecePBRTextures = [NSMutableArray arrayWithCapacity:7];
    
    for (int i = 0; i < 7; ++i) {
        MBCPiecePBRTextures *textures = [[MBCPiecePBRTextures alloc] init];
        [piecePBRTextures addObject:textures];
        if (i == 0) {
            // EMPTY entry for first item in MBCPieceCode
            continue;
        }
        
        NSString *pieceTypeName = pieceNamePrefixes[i - 1];
        
        NSString *textureName = [NSString stringWithFormat:@"%@N.jpg", pieceTypeName];
        textures.normalMap = [self loadTextureWithFileName:textureName
                                                     label:[textureName stringByDeletingPathExtension]
                                                    isSRGB:NO
                                           generateMipmaps:NO];
        
        textureName = [NSString stringWithFormat:@"%@R_AO.jpg", pieceTypeName];
        textures.roughnessAmbientOcclusion = [self loadTextureWithFileName:textureName 
                                                                     label:[textureName stringByDeletingPathExtension]
                                                                    isSRGB:NO
                                                           generateMipmaps:NO];
    }
    
    _piecePBRTextures = [[NSArray alloc] initWithArray:piecePBRTextures];
}

/*!
 @abstract loadTextureAttriburesFromFile:
 @discussion Loads the attributes dictionaries from plist for the current board style for both board and pieces.
*/
- (void)loadTextureAttriburesFromFile {
    NSString *styleDirectory = [@"Styles" stringByAppendingPathComponent:_styleName];
    NSString *path = [[NSBundle mainBundle] pathForResource:@"Board"
                                                     ofType:@"plist"
                                                inDirectory:styleDirectory];
    _boardAttributes = [NSDictionary dictionaryWithContentsOfFile:path];
       
    styleDirectory = [@"Styles" stringByAppendingPathComponent:_styleName];
    path = [[NSBundle mainBundle] pathForResource:@"Piece"
                                           ofType:@"plist"
                                      inDirectory:styleDirectory];
    _pieceAttributes = [NSDictionary dictionaryWithContentsOfFile:path];
}

/*!
 @abstract loadField:fromAttr:color:entry:
 @param field A pointer to the float to be updated
 @param attr Attribute dictionary used to read foat value
 @param color Index of color to use
 @param entry The string key value to read for setting the float
 @discussion Will update the pointed to float value in field with value read from attribute dictionary
*/
- (void)updateField:(float *)field
           fromAttr:(NSDictionary *)attr
              color:(MBCStyleColorIndex)color
              entry:(NSString *)entry {
    // Will get value by key that is created by appending entry to color, such as
    // Black+Diffuse, Black+Specular, Black+Shininess, Black+Alpha
    // See one of Board.plist or Piece.plist files in Resources/Styles
                
    NSString *colorString = (color == MBCStyleColorIndexWhite) ? @"White" : @"Black";
    NSNumber *val = [attr objectForKey:[colorString stringByAppendingString:entry]];
    if (val) {
        *field = [val floatValue];
    }
}

/*!
 @abstract loadTextureWithFileName:label:isSRGB:generateMipmaps:
 @param fileName File name of asset with extension
 @param label Debug label for textrue in metal frame capture
 @param isSRGB When YES, will load texture as sRGB color.
 @param generateMipmaps Pass YES to generate mipmaps for the texture
 @discussion Load the MTLTexture with the given name from the project's Images.xcassets
*/
- (id<MTLTexture>)loadTextureWithFileName:(NSString *)fileName label:(NSString *)label isSRGB:(BOOL)isSRGB generateMipmaps:(BOOL)generateMipmaps {
    NSDictionary *options = @{
        MTKTextureLoaderOptionGenerateMipmaps:      @(generateMipmaps),
        MTKTextureLoaderOptionTextureUsage:         @(MTLTextureUsageShaderRead),
        MTKTextureLoaderOptionTextureStorageMode:   @(MTLStorageModePrivate),
        MTKTextureLoaderOptionOrigin:               MTKTextureLoaderOriginBottomLeft,
        MTKTextureLoaderOptionSRGB:                 @(isSRGB)
    };

    NSURL* url = [[NSBundle mainBundle] URLForResource:fileName withExtension:@""];

    NSError *error;
    id <MTLTexture> texture = [_textureLoader newTextureWithContentsOfURL:url
                                                                  options:options
                                                                    error:&error];
    texture.label = label;
    
    if (error) {
        NSLog(@"Error loading texture from file %@: %@", fileName, error);
    }
    
    return texture;
}

/*!
 @abstract loadTextureWithName:
 @param name Name of the texture to load from Images.xcassets
 @discussion Load the MTLTexture with the given name from the project's Images.xcassets
*/
- (id<MTLTexture>)loadTextureWithName:(NSString *)name {
    if (!name) {
        return nil;
    }
    
    // Load the textures with shader read using private storage
    NSDictionary *textureLoaderOptions = @{
        MTKTextureLoaderOptionTextureUsage       : @(MTLTextureUsageShaderRead),
        MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModePrivate)
    };

    id<MTLTexture> texture = [_textureLoader newTextureWithName:name
                                                    scaleFactor:1.0
                                                         bundle:nil
                                                        options:textureLoaderOptions
                                                          error:nil];
    texture.label = name;
    
    return texture;
}

/*!
 @abstract textureNameForType:color:
 @param type The type of asset (ie board or piece)
 @param color The color if type is a piece
 @discussion Returns the texture name that can be used to load an MTLTexture from asset catalog
*/
- (NSString *)textureNameForType:(MBCMaterialType)type color:(MBCStyleColorIndex)color {
    
    if (type == MBCMaterialTypeBoard) {
        return [NSString stringWithFormat:@"Board%@", _styleName];
    } else if (type == MBCMaterialTypePiece) {
        NSString *colorPrefix = (color == MBCStyleColorIndexWhite) ? @"WhitePiece" : @"BlackPiece";
        return [NSString stringWithFormat:@"%@%@", colorPrefix, _styleName];
    }
    
    return nil;
}

- (id<MTLTexture>)blackPieceBaseColorTextureForCurrentStyle {
    NSString *name = [NSString stringWithFormat:@"BlackPiece%@.jpg", _styleName];
    return [self loadTextureWithFileName:name label:[name stringByDeletingPathExtension] isSRGB:YES generateMipmaps:YES];
}

- (id<MTLTexture>)whitePieceBaseColorTextureForCurrentStyle {
    NSString *name = [NSString stringWithFormat:@"WhitePiece%@.jpg", _styleName];
    return [self loadTextureWithFileName:name label:[name stringByDeletingPathExtension] isSRGB:YES generateMipmaps:YES];
}

- (id<MTLTexture>)boardTextureForCurrentStyle {
    NSString *name = [NSString stringWithFormat:@"Board%@.jpg", _styleName];
    return [self loadTextureWithFileName:name label:[name stringByDeletingPathExtension] isSRGB:YES generateMipmaps:YES];
}

- (void)updateAttributesForDrawStyle:(MBCDrawStyle *)drawStyle
                            forColor:(MBCStyleColorIndex)color
                                type:(MBCMaterialType)type
                                attr:(NSDictionary *)attr {
    // Update the numerical values for style from stored attribute dictionaries.
    [self updateField:&drawStyle->fDiffuse
          fromAttr:attr color:color entry:@"Diffuse"];
    [self updateField:&drawStyle->fSpecular
          fromAttr:attr color:color entry:@"Specular"];
    [self updateField:&drawStyle->fShininess
          fromAttr:attr color:color entry:@"Shininess"];
    [self updateField:&drawStyle->fAlpha
          fromAttr:attr color:color entry:@"Alpha"];
    
    [self updateField:&drawStyle->fMaterial.roughness
           fromAttr:attr color:color entry:@"Roughness"];
    [self updateField:&drawStyle->fMaterial.metallic
           fromAttr:attr color:color entry:@"Metallic"];
    [self updateField:&drawStyle->fMaterial.ambientOcclusion
           fromAttr:attr color:color entry:@"AmbientOcclusion"];
}

- (void)loadMaterialsForRendererID:(NSString *)rendererID newStyle:(NSString *)newStyle {
    BOOL loadStyle = (_boardDrawStyles[newStyle] == nil);
    
    if (loadStyle) {
        
        // Selects the directory location for corresponding style, such as "Styles/Wood" or "Styles/Marble"
        // and also used to build name of texture to load for board and pieces
        _styleName = newStyle;
        
        // Load stored numeric texture settings from plist file for current board and piece styles.
        [self loadTextureAttriburesFromFile];
        if (!_piecePBRTextures) {
            [self loadPiecePBRTextures];
        }
        
        // Intialize the materials associated with pieces.
        id<MTLTexture> baseColorBlack = [self blackPieceBaseColorTextureForCurrentStyle];
        id<MTLTexture> baseColorWhite = [self whitePieceBaseColorTextureForCurrentStyle];
        
        [self initializePieceStyleArraysForStyle:_styleName];
        
        NSArray<MBCDrawStyle *> *whitePieceDrawStyles = _whitePieceDrawStyles[_styleName];
        NSArray<MBCDrawStyle *> *blackPieceDrawStyles = _blackPieceDrawStyles[_styleName];
        
        for (int i = 1; i <= 6; ++i) {
            MBCDrawStyle *whitePieceStyle = whitePieceDrawStyles[i];
            [self updateAttributesForDrawStyle:whitePieceStyle
                                      forColor:MBCStyleColorIndexWhite
                                          type:MBCMaterialTypePiece
                                          attr:_pieceAttributes];
            whitePieceStyle.baseColorTexture = baseColorWhite;
            whitePieceStyle.normalMapTexture = _piecePBRTextures[i].normalMap;
            whitePieceStyle.roughnessAmbientOcclusionTexture = _piecePBRTextures[i].roughnessAmbientOcclusion;
            
            MBCDrawStyle *blackPieceStyle = blackPieceDrawStyles[i];
            [self updateAttributesForDrawStyle:blackPieceStyle
                                      forColor:MBCStyleColorIndexBlack
                                          type:MBCMaterialTypePiece
                                          attr:_pieceAttributes];
            blackPieceStyle.baseColorTexture = baseColorBlack;
            blackPieceStyle.normalMapTexture = _piecePBRTextures[i].normalMap;
            blackPieceStyle.roughnessAmbientOcclusionTexture = _piecePBRTextures[i].roughnessAmbientOcclusion;
        }
        
        // Initialize the material for the selected promotion piece.
        (void)[_selectedPieceDrawStyle initWithTexture:0];
        _selectedPieceDrawStyle->fAlpha = 0.8f;
        
        // Initialize the material associated with the board.
        MBCDrawStyle *boardDrawStyle = [[MBCDrawStyle alloc] init];
        _boardDrawStyles[_styleName] = boardDrawStyle;
        
        [self updateAttributesForDrawStyle:boardDrawStyle
                                  forColor:MBCStyleColorIndexWhite
                                      type:MBCMaterialTypeBoard
                                      attr:_boardAttributes];
        boardDrawStyle.baseColorTexture = [self boardTextureForCurrentStyle];
        boardDrawStyle.normalMapTexture = nil;
        boardDrawStyle.roughnessAmbientOcclusionTexture = nil;
        
        // May not be needed, but will keep for now.
        [self updateField:&_boardReflectivity
                 fromAttr:_boardAttributes
                    color:MBCStyleColorIndexWhite
                    entry:@"Reflectivity"];
        
        // Done using this style in loading methods
        _styleName = nil;
    }
    
    // Update the usage for the style.
    [_styleUsageMap[newStyle] addObject:rendererID];
}

- (void)releaseUsageForStyle:(NSString *)style rendererID:(NSString *)rendererID {
    [_styleUsageMap[style] removeObject:rendererID];
    if (_styleUsageMap[style].count == 0) {
        _boardDrawStyles[style] = nil;
        _whitePieceDrawStyles[style] = nil;
        _blackPieceDrawStyles[style] = nil;
    }
}

- (MBCDrawStyle *)materialForPiece:(MBCPiece)piece style:(NSString *)style {
    MBCPieceCode color = Color(piece);
    MBCPieceCode pieceType = Piece(piece);
    NSAssert(pieceType > 0 && pieceType <= 6, @"Piece type must be one of MBCPieceCode: KING, QUEEN, BISHOP, KNIGHT, ROOK, PAWN");
    if (color == kWhitePiece) {
        return _whitePieceDrawStyles[style][pieceType];
    } else {
        return _blackPieceDrawStyles[style][pieceType];
    }
}

- (MBCDrawStyle *)boardMaterialWithStyle:(NSString *)style {
    return _boardDrawStyles[style];
}

@end
