/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#ifndef FTTRIGON_H_
#define FTTRIGON_H_

#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *  computations
   *
   */


  /**************************************************************************
   *
   * @type:
   *   FT_Angle
   *
   * @description:
   *   This type is used to model angle values in FreeType.  Note that the
   *   angle is a 16.16 fixed-point value expressed in degrees.
   *
   */
  typedef FT_Fixed  FT_Angle;


  /**************************************************************************
   *
   * @macro:
   *   FT_ANGLE_PI
   *
   * @description:
   *   The angle pi expressed in @FT_Angle units.
   *
   */
#define FT_ANGLE_PI  ( 180L << 16 )


  /**************************************************************************
   *
   * @macro:
   *   FT_ANGLE_2PI
   *
   * @description:
   *   The angle 2*pi expressed in @FT_Angle units.
   *
   */
#define FT_ANGLE_2PI  ( FT_ANGLE_PI * 2 )


  /**************************************************************************
   *
   * @macro:
   *   FT_ANGLE_PI2
   *
   * @description:
   *   The angle pi/2 expressed in @FT_Angle units.
   *
   */
#define FT_ANGLE_PI2  ( FT_ANGLE_PI / 2 )


  /**************************************************************************
   *
   * @macro:
   *   FT_ANGLE_PI4
   *
   * @description:
   *   The angle pi/4 expressed in @FT_Angle units.
   *
   */
#define FT_ANGLE_PI4  ( FT_ANGLE_PI / 4 )


  /**************************************************************************
   *
   * @function:
   *   FT_Sin
   *
   * @description:
   *   Return the sinus of a given angle in fixed-point format.
   *
   * @input:
   *   angle ::
   *     The input angle.
   *
   * @return:
   *   The sinus value.
   *
   * @note:
   *   If you need both the sinus and cosinus for a given angle, use the
   *   function @FT_Vector_Unit.
   *
   */
  FT_EXPORT( FT_Fixed )
  FT_Sin( FT_Angle  angle );


  /**************************************************************************
   *
   * @function:
   *   FT_Cos
   *
   * @description:
   *   Return the cosinus of a given angle in fixed-point format.
   *
   * @input:
   *   angle ::
   *     The input angle.
   *
   * @return:
   *   The cosinus value.
   *
   * @note:
   *   If you need both the sinus and cosinus for a given angle, use the
   *   function @FT_Vector_Unit.
   *
   */
  FT_EXPORT( FT_Fixed )
  FT_Cos( FT_Angle  angle );


  /**************************************************************************
   *
   * @function:
   *   FT_Tan
   *
   * @description:
   *   Return the tangent of a given angle in fixed-point format.
   *
   * @input:
   *   angle ::
   *     The input angle.
   *
   * @return:
   *   The tangent value.
   *
   */
  FT_EXPORT( FT_Fixed )
  FT_Tan( FT_Angle  angle );


  /**************************************************************************
   *
   * @function:
   *   FT_Atan2
   *
   * @description:
   *   Return the arc-tangent corresponding to a given vector (x,y) in the 2d
   *   plane.
   *
   * @input:
   *   x ::
   *     The horizontal vector coordinate.
   *
   *   y ::
   *     The vertical vector coordinate.
   *
   * @return:
   *   The arc-tangent value (i.e. angle).
   *
   */
  FT_EXPORT( FT_Angle )
  FT_Atan2( FT_Fixed  x,
            FT_Fixed  y );


  /**************************************************************************
   *
   * @function:
   *   FT_Angle_Diff
   *
   * @description:
   *   Return the difference between two angles.  The result is always
   *   constrained to the ]-PI..PI] interval.
   *
   * @input:
   *   angle1 ::
   *     First angle.
   *
   *   angle2 ::
   *     Second angle.
   *
   * @return:
   *   Constrained value of `angle2-angle1`.
   *
   */
  FT_EXPORT( FT_Angle )
  FT_Angle_Diff( FT_Angle  angle1,
                 FT_Angle  angle2 );


  /**************************************************************************
   *
   * @function:
   *   FT_Vector_Unit
   *
   * @description:
   *   Return the unit vector corresponding to a given angle.  After the
   *   call, the value of `vec.x` will be `cos(angle)`, and the value of
   *   `vec.y` will be `sin(angle)`.
   *
   *   This function is useful to retrieve both the sinus and cosinus of a
   *   given angle quickly.
   *
   * @output:
   *   vec ::
   *     The address of target vector.
   *
   * @input:
   *   angle ::
   *     The input angle.
   *
   */
  FT_EXPORT( void )
  FT_Vector_Unit( FT_Vector*  vec,
                  FT_Angle    angle );


  /**************************************************************************
   *
   * @function:
   *   FT_Vector_Rotate
   *
   * @description:
   *   Rotate a vector by a given angle.
   *
   * @inout:
   *   vec ::
   *     The address of target vector.
   *
   * @input:
   *   angle ::
   *     The input angle.
   *
   */
  FT_EXPORT( void )
  FT_Vector_Rotate( FT_Vector*  vec,
                    FT_Angle    angle );


  /**************************************************************************
   *
   * @function:
   *   FT_Vector_Length
   *
   * @description:
   *   Return the length of a given vector.
   *
   * @input:
   *   vec ::
   *     The address of target vector.
   *
   * @return:
   *   The vector length, expressed in the same units that the original
   *   vector coordinates.
   *
   */
  FT_EXPORT( FT_Fixed )
  FT_Vector_Length( FT_Vector*  vec );


  /**************************************************************************
   *
   * @function:
   *   FT_Vector_Polarize
   *
   * @description:
   *   Compute both the length and angle of a given vector.
   *
   * @input:
   *   vec ::
   *     The address of source vector.
   *
   * @output:
   *   length ::
   *     The vector length.
   *
   *   angle ::
   *     The vector angle.
   *
   */
  FT_EXPORT( void )
  FT_Vector_Polarize( FT_Vector*  vec,
                      FT_Fixed   *length,
                      FT_Angle   *angle );


  /**************************************************************************
   *
   * @function:
   *   FT_Vector_From_Polar
   *
   * @description:
   *   Compute vector coordinates from a length and angle.
   *
   * @output:
   *   vec ::
   *     The address of source vector.
   *
   * @input:
   *   length ::
   *     The vector length.
   *
   *   angle ::
   *     The vector angle.
   *
   */
  FT_EXPORT( void )
  FT_Vector_From_Polar( FT_Vector*  vec,
                        FT_Fixed    length,
                        FT_Angle    angle );

  /* */


FT_END_HEADER

#endif /* FTTRIGON_H_ */


/* END */
