/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
/*!
 * @header IOAudioBlitterLibDispatch
 */

#ifndef __IOAudioBlitterLibDispatch_h__
#define __IOAudioBlitterLibDispatch_h__

#include <libkern/OSTypes.h>


#pragma mark -
#pragma mark Alternate names

// Alternate names for the functions below: these explicitly specify the endianness of the integer format instead of "native"/"swap"
#if TARGET_RT_BIG_ENDIAN
#define	IOAF_BEInt16ToFloat32	IOAF_NativeInt16ToFloat32
#define	IOAF_LEInt16ToFloat32	IOAF_SwapInt16ToFloat32
#define	IOAF_BEInt24ToFloat32	IOAF_NativeInt24ToFloat32
#define	IOAF_LEInt24ToFloat32	IOAF_SwapInt24ToFloat32
#define	IOAF_BEInt32ToFloat32	IOAF_NativeInt32ToFloat32
#define	IOAF_LEInt32ToFloat32	IOAF_SwapInt32ToFloat32

#define IOAF_Float32ToBEInt16	IOAF_Float32ToNativeInt16
#define IOAF_Float32ToLEInt16	IOAF_Float32ToSwapInt16
#define IOAF_Float32ToBEInt24	IOAF_Float32ToNativeInt24
#define IOAF_Float32ToLEInt24	IOAF_Float32ToSwapInt24
#define IOAF_Float32ToBEInt32	IOAF_Float32ToNativeInt32
#define IOAF_Float32ToLEInt32	IOAF_Float32ToSwapInt32
#else
#define	IOAF_LEInt16ToFloat32	IOAF_NativeInt16ToFloat32
#define	IOAF_BEInt16ToFloat32	IOAF_SwapInt16ToFloat32
#define	IOAF_LEInt24ToFloat32	IOAF_NativeInt24ToFloat32
#define	IOAF_BEInt24ToFloat32	IOAF_SwapInt24ToFloat32
#define	IOAF_LEInt32ToFloat32	IOAF_NativeInt32ToFloat32
#define	IOAF_BEInt32ToFloat32	IOAF_SwapInt32ToFloat32

#define IOAF_Float32ToLEInt16	IOAF_Float32ToNativeInt16
#define IOAF_Float32ToBEInt16	IOAF_Float32ToSwapInt16
#define IOAF_Float32ToLEInt24	IOAF_Float32ToNativeInt24
#define IOAF_Float32ToBEInt24	IOAF_Float32ToSwapInt24
#define IOAF_Float32ToLEInt32	IOAF_Float32ToNativeInt32
#define IOAF_Float32ToBEInt32	IOAF_Float32ToSwapInt32
#endif

/*!
 * @typedef Float32
 * @abstract Convenience type that represent a 32-bit floating point number
 */
typedef float	Float32;

/*!
 * @typedef Float64
 * @abstract Convenience type that represent a 64-bit floating point number
 */
typedef double	Float64;


/*!
 * @function IOAF_NativeInt16ToFloat32
 * @abstract Converts native 16-bit integer float to 32-bit float
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_NativeInt16ToFloat32( const SInt16 *src, Float32 *dest, unsigned int count );

/*!
 * @function IOAF_SwapInt16ToFloat32
 * @abstract Converts non-native 16-bit integer float to 32-bit float
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_SwapInt16ToFloat32( const SInt16 *src, Float32 *dest, unsigned int count );

/*!
 * @function IOAF_NativeInt24ToFloat32
 * @abstract Converts native 24-bit integer float to 32-bit float
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_NativeInt24ToFloat32( const UInt8 *src, Float32 *dest, unsigned int count );

/*!
 * @function IOAF_SwapInt24ToFloat32
 * @abstract Converts non-native 24-bit integer float to 32-bit float
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_SwapInt24ToFloat32( const UInt8 *src, Float32 *dest, unsigned int count );

/*!
 * @function IOAF_NativeInt32ToFloat32
 * @abstract Converts native 32-bit integer float to 32-bit float
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_NativeInt32ToFloat32( const SInt32 *src, Float32 *dest, unsigned int count );

/*!
 * @function IOAF_SwapInt32ToFloat32
 * @abstract Converts non-native 32-bit integer float to 32-bit float
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_SwapInt32ToFloat32( const SInt32 *src, Float32 *dest, unsigned int count );


/*!
 * @function IOAF_Float32ToNativeInt16
 * @abstract Converts 32-bit floating point to native 16-bit integer
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_Float32ToNativeInt16( const Float32 *src, SInt16 *dest, unsigned int count );

/*!
 * @function IOAF_Float32ToSwapInt16
 * @abstract Converts 32-bit floating point to non-native 16-bit integer
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_Float32ToSwapInt16( const Float32 *src, SInt16 *dest, unsigned int count );

/*!
 * @function IOAF_Float32ToNativeInt24
 * @abstract Converts 32-bit floating point to native 24-bit integer
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_Float32ToNativeInt24( const Float32 *src, UInt8 *dest, unsigned int count );

/*!
 * @function IOAF_Float32ToSwapInt24
 * @abstract Converts 32-bit floating point to non-native 24-bit integer
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_Float32ToSwapInt24( const Float32 *src, UInt8 *dest, unsigned int count );

/*!
 * @function IOAF_Float32ToNativeInt32
 * @abstract Converts 32-bit floating point to native 32-bit integer
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_Float32ToNativeInt32( const Float32 *src, SInt32 *dest, unsigned int count );

/*!
 * @function IOAF_Float32ToSwapInt32
 * @abstract Converts 32-bit floating point to non-native 32-bit integer
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_Float32ToSwapInt32( const Float32 *src, SInt32 *dest, unsigned int count );

/*!
 * @function IOAF_bcopy_WriteCombine
 * @abstract An efficient bcopy from "write combine" memory to regular memory. It is safe to assume that all memory has been copied when the function has completed
 * @param src Pointer to the data to convert
 * @param dest Pointer to the converted data
 * @param count The number of items to convert
 */
extern void IOAF_bcopy_WriteCombine(const void *src, void *dest, unsigned int count );

#endif // __IOAudioBlitterLibDispatch_h__
