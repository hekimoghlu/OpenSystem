/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
#include "config.h"
#include "FuzzerPredictions.h"

#include <wtf/text/StringToIntegerConversion.h>

namespace JSC {

static String readFileIntoString(const char* fileName)
{
    FILE* file = fopen(fileName, "r");
    RELEASE_ASSERT_WITH_MESSAGE(file, "Failed to open file %s", fileName);
    RELEASE_ASSERT(fseek(file, 0, SEEK_END) != -1);
    long bufferCapacity = ftell(file);
    RELEASE_ASSERT(bufferCapacity != -1);
    RELEASE_ASSERT(fseek(file, 0, SEEK_SET) != -1);

    std::span<LChar> buffer;
    String string = String::createUninitialized(bufferCapacity, buffer);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    size_t readSize = fread(buffer.data(), 1, buffer.size(), file);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    fclose(file);
    RELEASE_ASSERT(readSize == static_cast<size_t>(bufferCapacity));
    return string;
}

FuzzerPredictions::FuzzerPredictions(const char* filename)
{
    RELEASE_ASSERT_WITH_MESSAGE(filename, "prediction file must be specified using --fuzzerPredictionsFile=");

    String predictions = readFileIntoString(filename);
    const Vector<String>& lines = predictions.split('\n');
    for (const auto& line : lines) {
        // Predictions are stored in a text file, one prediction per line in the colon delimited format:
        // <lookup key>:<prediction in hex without leading 0x>
        // The lookup key is a pipe separated string with the format:
        // <filename>|<opcode>|<start offset>|<end offset>

        // The start and end offsets are 7-bit unsigned integers.
        // If start offset > 127, then both start and end offsets are 0.
        // If end offset > 127, then the end offset is 0.

        // Example predictions:
        // foo.js|op_construct|702|721:1000084
        // foo.js|op_call|748|760:408800

        // Predictions can be generated using PredictionFileCreatingFuzzerAgent.
        // Some opcodes are aliased together to make generating the predictions more straightforward.
        // For the aliases see: FileBasedFuzzerAgentBase::opcodeAliasForLookupKey()

        // FIXME: The current implementation only supports one prediction per lookup key.

        const Vector<String>& lineParts = line.split(':');
        RELEASE_ASSERT_WITH_MESSAGE(lineParts.size() == 2, "Expected line with two parts delimited by a colon. Found line with %zu parts.", lineParts.size());
        const String& lookupKey = lineParts[0];
        const String& predictionString = lineParts[1];
        auto prediction = parseInteger<uint64_t>(predictionString, 0x10);
        RELEASE_ASSERT_WITH_MESSAGE(prediction, "Could not parse prediction from '%s'", predictionString.utf8().data());
        RELEASE_ASSERT(speculationChecked(*prediction, SpecFullTop));
        m_predictions.set(lookupKey, *prediction);
    }
}

std::optional<SpeculatedType> FuzzerPredictions::predictionFor(const String& key)
{
    auto it = m_predictions.find(key);
    if (it == m_predictions.end())
        return std::nullopt;
    return it->value;
}

FuzzerPredictions& ensureGlobalFuzzerPredictions()
{
    static LazyNeverDestroyed<FuzzerPredictions> fuzzerPredictions;
    static std::once_flag initializeFuzzerPredictionsFlag;
    std::call_once(initializeFuzzerPredictionsFlag, [] {
        const char* fuzzerPredictionsFilename = Options::fuzzerPredictionsFile();
        fuzzerPredictions.construct(fuzzerPredictionsFilename);
    });
    return fuzzerPredictions;
}

} // namespace JSC
