/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#pragma once

// Defines the interface for several vector math functions whose implementation will ideally be optimized.

namespace WebCore {

namespace VectorMath {

// Multiples inputVector by scalar then adds the result to outputVector (simplified vsma).
// for (n = 0; n < numberOfElementsToProcess; ++n)
//     outputVector[n] += inputVector[n] * scalar;
void multiplyByScalarThenAddToOutput(std::span<const float> inputVector, float scalar, std::span<float> outputVector);

// Adds a vector inputVector2 to the product of a scalar value and a single-precision vector inputVector1 (vsma).
// for (n = 0; n < numberOfElementsToProcess; ++n)
//     outputVector[n] = inputVector1[n] * scalar + inputVector2[n];
void multiplyByScalarThenAddToVector(std::span<const float> inputVector1, float scalar, std::span<const float> inputVector2, std::span<float> outputVector);

// Multiplies the sum of two vectors by a scalar value (vasm).
void addVectorsThenMultiplyByScalar(std::span<const float> inputVector1, std::span<const float> inputVector2, float scalar, std::span<float> outputVector);

void multiplyByScalar(std::span<const float> inputVector, float scalar, std::span<float> outputVector);
void addScalar(std::span<const float> inputVector, float scalar, std::span<float> outputVector);
void substract(std::span<const float> inputVector1, std::span<const float> inputVector2, std::span<float> outputVector);

void add(std::span<const int> inputVector1, std::span<const int> inputVector2, std::span<int> outputVector);
void add(std::span<const float> inputVector1, std::span<const float> inputVector2, std::span<float> outputVector);
void add(std::span<const double> inputVector1, std::span<const double> inputVector2, std::span<double> outputVector);

// result = sum(inputVector1[n] * inputVector2[n], 0 <= n < inputVector1.size());
float dotProduct(std::span<const float> inputVector1, std::span<const float> inputVector2);

// Finds the maximum magnitude of a float vector.
float maximumMagnitude(std::span<const float> inputVector);

// Sums the squares of a float vector's elements (svesq).
float sumOfSquares(std::span<const float> inputVector);

// For an element-by-element multiply of two float vectors.
void multiply(std::span<const float> inputVector1, std::span<const float> inputVector2, std::span<float> outputVector);

// Multiplies two complex vectors (zvmul).
void multiplyComplex(std::span<const float> realVector1, std::span<const float> imagVector1, std::span<const float> realVector2, std::span<const float> imagVector2, std::span<float> realOutputVector, std::span<float> imagOutputVector);

// Copies elements while clipping values to the threshold inputs.
void clamp(std::span<const float> inputVector, float mininum, float maximum, std::span<float> outputVector);

void linearToDecibels(std::span<const float> inputVector, std::span<float> outputVector);

// Calculates the linear interpolation between the supplied single-precision vectors
// for (n = 0; n < numberOfElementsToProcess; ++n)
//     outputVector[n] = inputVector1[n] + interpolationFactor * (inputVector2[n] - inputVector1[n]);
// NOTE: Internal implementation may modify inputVector2.
void interpolate(std::span<const float> inputVector1, std::span<float> inputVector2, float interpolationFactor, std::span<float> outputVector);

} // namespace VectorMath

} // namespace WebCore
