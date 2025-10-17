/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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
#include <cstddef>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace {

constexpr uintptr_t minAlignShift = 4;
constexpr uintptr_t minAlign = static_cast<uintptr_t>(1) << minAlignShift;
constexpr uintptr_t upperBoundExclusive = 1000;
constexpr uintptr_t sizeCellColumns = 4;

void printSizeCell(uintptr_t size)
{
    ostringstream sizeStringOut;
    sizeStringOut << size;

    string sizeString = sizeStringOut.str();
    while (sizeString.size() < sizeCellColumns)
        sizeString = " " + sizeString;

    cout << sizeString;
}

void printProgression(const string& name, function<uintptr_t(uintptr_t)> mapping)
{
    cout << name << "\n";
    
    cout << "    ";
    for (uintptr_t size = 0; size < upperBoundExclusive; size += minAlign)
        printSizeCell(size);
    cout << "\n";
    
    cout << "    ";
    for (uintptr_t size = 0; size < upperBoundExclusive; size += minAlign)
        printSizeCell(mapping(size));
    cout << "\n";
}

#define PRINT_PROGRESSION(mapping) \
    printProgression(#mapping, (mapping))

} // anonymous namespace

int main(int argc, char** argv)
{
    PRINT_PROGRESSION([] (uintptr_t size) { return (size + minAlign - 1) >> minAlignShift; });
    PRINT_PROGRESSION([] (uintptr_t size) { return ((size + minAlign - 1) >> minAlignShift) - size * size / 14000; });
    PRINT_PROGRESSION([] (uintptr_t size) {
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          return index - index * index / 64;
                      });
    PRINT_PROGRESSION([] (uintptr_t size) {
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          return index - index * index / 48;
                      });
    PRINT_PROGRESSION([] (uintptr_t size) {
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          if (index <= 6)
                              return index;
                          return index - index * index / 48;
                      });
    PRINT_PROGRESSION([] (uintptr_t size) {
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          if (index <= 6)
                              return index;
                          return index - index * index / 40;
                      });
    PRINT_PROGRESSION([] (uintptr_t size) {
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          return index - index * index / 32;
                      });
    PRINT_PROGRESSION([] (uintptr_t size) {
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          if (index <= 5)
                              return index;
                          return index - index * index / 32;
                      });
    PRINT_PROGRESSION([] (uintptr_t size) {
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          if (index <= 5)
                              return index;
                          uintptr_t square = index * index;
                          if (index <= 14)
                              return index - square / 32;
                          return index - square / 50 - 2;
                      });
    PRINT_PROGRESSION([] (uintptr_t size) {
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          if (index <= 5)
                              return index;
                          if (index <= 12)
                              return ((index - 1) >> 1) + 3;
                          return (index >> 2) + 6;
                      });
    PRINT_PROGRESSION([] (uintptr_t size) {
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          if (index <= 6)
                              return index;
                          if (index <= 12)
                              return ((index - 1) >> 1) + 4;
                          return (index >> 2) + 7;
                      });
    PRINT_PROGRESSION([] (uintptr_t size) -> uintptr_t {
                          if (!size)
                              return 0;
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          if (index <= 5)
                              return index - 1;
                          if (index <= 12)
                              return ((index - 1) >> 1) + 2;
                          return (index >> 2) + 5;
                      });
    PRINT_PROGRESSION([] (uintptr_t size) -> uintptr_t {
                          if (!size)
                              return 0;
                          uintptr_t index = (size + minAlign - 1) >> minAlignShift;
                          if (index <= 6)
                              return index - 1;
                          if (index <= 12)
                              return ((index - 1) >> 1) + 3;
                          return (index >> 2) + 6;
                      });
    
    return 0;
}

