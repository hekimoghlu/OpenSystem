/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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

// Â© 2018 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

// simplenormperf.cpp
// created: 2018mar15 Markus W. Scherer

#include <stdio.h>
#include <string>

#include "unicode/utypes.h"
#include "unicode/bytestream.h"
#include "unicode/normalizer2.h"
#include "unicode/stringpiece.h"
#include "unicode/unistr.h"
#include "unicode/utf8.h"
#include "unicode/utimer.h"
#include "cmemory.h"

using icu::Normalizer2;
using icu::UnicodeString;

namespace {

// Strings with commonly occurring BMP characters.
class CommonChars {
public:
    static UnicodeString getMixed(int32_t minLength) {
        return extend(UnicodeString(latin1).append(japanese).append(arabic), minLength);
    }
    static UnicodeString getLatin1(int32_t minLength) { return extend(latin1, minLength); }
    static UnicodeString getLowercaseLatin1(int32_t minLength) { return extend(lowercaseLatin1, minLength); }
    static UnicodeString getASCII(int32_t minLength) { return extend(ascii, minLength); }
    static UnicodeString getJapanese(int32_t minLength) { return extend(japanese, minLength); }

    // Returns an array of UTF-8 offsets, one per code point.
    // Assumes all BMP characters.
    static int32_t *toUTF8WithOffsets(const UnicodeString &s16, std::string &s8, int32_t &numCodePoints) {
        s8.clear();
        s8.reserve(s16.length());
        s16.toUTF8String(s8);
        const char *s = s8.data();
        int32_t length = s8.length();
        int32_t *offsets = new int32_t[length + 1];
        int32_t numCP = 0;
        for (int32_t i = 0; i < length;) {
            offsets[numCP++] = i;
            U8_FWD_1(s, i, length);
        }
        offsets[numCP] = length;
        numCodePoints = numCP;
        return offsets;
    }

private:
    static UnicodeString extend(const UnicodeString &s, int32_t minLength) {
        UnicodeString result(s);
        while (result.length() < minLength) {
            UnicodeString twice = result + result;
            result = std::move(twice);
        }
        return result;
    }

    static const char16_t *const latin1;
    static const char16_t *const lowercaseLatin1;
    static const char16_t *const ascii;
    static const char16_t *const japanese;
    static const char16_t *const arabic;
};

const char16_t *const CommonChars::latin1 =
      // Goetheâ€™s BergschloÃŸ in normal sentence case.
      u"Da droben auf jenem Berge, da steht ein altes SchloÃŸ, "
      u"wo hinter Toren und TÃ¼ren sonst lauerten Ritter und RoÃŸ.\n"
      u"Verbrannt sind TÃ¼ren und Tore, und Ã¼berall ist es so still; "
      u"das alte verfallne GemÃ¤uer durchklettr ich, wie ich nur will.\n"
      u"Hierneben lag ein Keller, so voll von kÃ¶stlichem Wein; "
      u"nun steiget nicht mehr mit KrÃ¼gen die Kellnerin heiter hinein.\n"
      u"Sie setzt den GÃ¤sten im Saale nicht mehr die Becher umher, "
      u"sie fÃ¼llt zum Heiligen Mahle dem Pfaffen das FlÃ¤schchen nicht mehr.\n"
      u"Sie reicht dem lÃ¼sternen Knappen nicht mehr auf dem Gange den Trank, "
      u"und nimmt fÃ¼r flÃ¼chtige Gabe nicht mehr den flÃ¼chtigen Dank.\n"
      u"Denn alle Balken und Decken, sie sind schon lange verbrannt, "
      u"und Trepp und Gang und Kapelle in Schutt und TrÃ¼mmer verwandt.\n"
      u"Doch als mit Zither und Flasche nach diesen felsigen HÃ¶hn "
      u"ich an dem heitersten Tage mein Liebchen steigen gesehn,\n"
      u"da drÃ¤ngte sich frohes Behagen hervor aus verÃ¶deter Ruh, "
      u"da gings wie in alten Tagen recht feierlich wieder zu.\n"
      u"Als wÃ¤ren fÃ¼r stattliche GÃ¤ste die weitesten RÃ¤ume bereit, "
      u"als kÃ¤m ein PÃ¤rchen gegangen aus jener tÃ¼chtigen Zeit.\n"
      u"Als stÃ¼nd in seiner Kapelle der wÃ¼rdige Pfaffe schon da "
      u"und fragte: Wollt ihr einander? Wir aber lÃ¤chelten: Ja!\n"
      u"Und tief bewegten GesÃ¤nge des Herzens innigsten Grund, "
      u"Es zeugte, statt der Menge, der Echo schallender Mund.\n"
      u"Und als sich gegen Abend im stillen alles verlor,"
      u"da blickte die glÃ¼hende Sonne zum schroffen Gipfel empor.\n"
      u"Und Knapp und Kellnerin glÃ¤nzen als Herren weit und breit; "
      u"sie nimmt sich zum Kredenzen und er zum Danke sich Zeit.\n";

const char16_t *const CommonChars::lowercaseLatin1 =
      // Goetheâ€™s BergschloÃŸ in all lowercase
      u"da droben auf jenem berge, da steht ein altes schloÃŸ, "
      u"wo hinter toren und tÃ¼ren sonst lauerten ritter und roÃŸ.\n"
      u"verbrannt sind tÃ¼ren und tore, und Ã¼berall ist es so still; "
      u"das alte verfallne gemÃ¤uer durchklettr ich, wie ich nur will.\n"
      u"hierneben lag ein keller, so voll von kÃ¶stlichem wein; "
      u"nun steiget nicht mehr mit krÃ¼gen die kellnerin heiter hinein.\n"
      u"sie setzt den gÃ¤sten im saale nicht mehr die becher umher, "
      u"sie fÃ¼llt zum heiligen mahle dem pfaffen das flÃ¤schchen nicht mehr.\n"
      u"sie reicht dem lÃ¼sternen knappen nicht mehr auf dem gange den trank, "
      u"und nimmt fÃ¼r flÃ¼chtige gabe nicht mehr den flÃ¼chtigen dank.\n"
      u"denn alle balken und decken, sie sind schon lange verbrannt, "
      u"und trepp und gang und kapelle in schutt und trÃ¼mmer verwandt.\n"
      u"doch als mit zither und flasche nach diesen felsigen hÃ¶hn "
      u"ich an dem heitersten tage mein liebchen steigen gesehn,\n"
      u"da drÃ¤ngte sich frohes behagen hervor aus verÃ¶deter ruh, "
      u"da gings wie in alten tagen recht feierlich wieder zu.\n"
      u"als wÃ¤ren fÃ¼r stattliche gÃ¤ste die weitesten rÃ¤ume bereit, "
      u"als kÃ¤m ein pÃ¤rchen gegangen aus jener tÃ¼chtigen zeit.\n"
      u"als stÃ¼nd in seiner kapelle der wÃ¼rdige pfaffe schon da "
      u"und fragte: wollt ihr einander? wir aber lÃ¤chelten: ja!\n"
      u"und tief bewegten gesÃ¤nge des herzens innigsten grund, "
      u"es zeugte, statt der menge, der echo schallender mund.\n"
      u"und als sich gegen abend im stillen alles verlor,"
      u"da blickte die glÃ¼hende sonne zum schroffen gipfel empor.\n"
      u"und knapp und kellnerin glÃ¤nzen als herren weit und breit; "
      u"sie nimmt sich zum kredenzen und er zum danke sich zeit.\n";

const char16_t *const CommonChars::ascii =
      // Goetheâ€™s BergschloÃŸ in normal sentence case but ASCII-fied
      u"Da droben auf jenem Berge, da steht ein altes Schloss, "
      u"wo hinter Toren und Tueren sonst lauerten Ritter und Ross.\n"
      u"Verbrannt sind Tueren und Tore, und ueberall ist es so still; "
      u"das alte verfallne Gemaeuer durchklettr ich, wie ich nur will.\n"
      u"Hierneben lag ein Keller, so voll von koestlichem Wein; "
      u"nun steiget nicht mehr mit Kruegen die Kellnerin heiter hinein.\n"
      u"Sie setzt den Gaesten im Saale nicht mehr die Becher umher, "
      u"sie fuellt zum Heiligen Mahle dem Pfaffen das Flaeschchen nicht mehr.\n"
      u"Sie reicht dem luesternen Knappen nicht mehr auf dem Gange den Trank, "
      u"und nimmt fuer fluechtige Gabe nicht mehr den fluechtigen Dank.\n"
      u"Denn alle Balken und Decken, sie sind schon lange verbrannt, "
      u"und Trepp und Gang und Kapelle in Schutt und Truemmer verwandt.\n"
      u"Doch als mit Zither und Flasche nach diesen felsigen Hoehn "
      u"ich an dem heitersten Tage mein Liebchen steigen gesehn,\n"
      u"da draengte sich frohes Behagen hervor aus veroedeter Ruh, "
      u"da gings wie in alten Tagen recht feierlich wieder zu.\n"
      u"Als waeren fuer stattliche Gaeste die weitesten Raeume bereit, "
      u"als kaem ein Paerchen gegangen aus jener tuechtigen Zeit.\n"
      u"Als stuend in seiner Kapelle der wuerdige Pfaffe schon da "
      u"und fragte: Wollt ihr einander? Wir aber laechelten: Ja!\n"
      u"Und tief bewegten Gesaenge des Herzens innigsten Grund, "
      u"Es zeugte, statt der Menge, der Echo schallender Mund.\n"
      u"Und als sich gegen Abend im stillen alles verlor,"
      u"da blickte die gluehende Sonne zum schroffen Gipfel empor.\n"
      u"Und Knapp und Kellnerin glaenzen als Herren weit und breit; "
      u"sie nimmt sich zum Kredenzen und er zum Danke sich Zeit.\n";

const char16_t *const CommonChars::japanese =
      // Ame ni mo makezu = Be not Defeated by the Rain, by Kenji Miyazawa.
      u"é›¨ã«ã‚‚ã¾ã‘ãšé¢¨ã«ã‚‚ã¾ã‘ãšé›ªã«ã‚‚å¤ã®æš‘ã•ã«ã‚‚ã¾ã‘ã¬"
      u"ä¸ˆå¤«ãªã‹ã‚‰ã ã‚’ã‚‚ã¡æ…¾ã¯ãªãæ±ºã—ã¦çž‹ã‚‰ãš"
      u"ã„ã¤ã‚‚ã—ãšã‹ã«ã‚ã‚‰ã£ã¦ã„ã‚‹ä¸€æ—¥ã«çŽ„ç±³å››åˆã¨"
      u"å‘³å™Œã¨å°‘ã—ã®é‡Žèœã‚’ãŸã¹ã‚ã‚‰ã‚†ã‚‹ã“ã¨ã‚’"
      u"ã˜ã¶ã‚“ã‚’ã‹ã‚“ã˜ã‚‡ã†ã«ã„ã‚Œãšã«ã‚ˆãã¿ããã—ã‚ã‹ã‚Š"
      u"ãã—ã¦ã‚ã™ã‚Œãšé‡ŽåŽŸã®æ¾ã®æž—ã®è”­ã®"
      u"å°ã•ãªè±ã¶ãã®å°å±‹ã«ã„ã¦æ±ã«ç—…æ°—ã®ã“ã©ã‚‚ã‚ã‚Œã°"
      u"è¡Œã£ã¦çœ‹ç—…ã—ã¦ã‚„ã‚Šè¥¿ã«ã¤ã‹ã‚ŒãŸæ¯ã‚ã‚Œã°"
      u"è¡Œã£ã¦ãã®ç¨²ã®æŸã‚’è² ã„å—ã«æ­»ã«ãã†ãªäººã‚ã‚Œã°"
      u"è¡Œã£ã¦ã“ã‚ãŒã‚‰ãªãã¦ã‚‚ã„ã„ã¨ã„ã„"
      u"åŒ—ã«ã‘ã‚“ã‹ã‚„ãã—ã‚‡ã†ãŒã‚ã‚Œã°"
      u"ã¤ã¾ã‚‰ãªã„ã‹ã‚‰ã‚„ã‚ã‚ã¨ã„ã„ã²ã§ã‚Šã®ã¨ãã¯ãªã¿ã ã‚’ãªãŒã—"
      u"ã•ã‚€ã•ã®ãªã¤ã¯ãŠã‚ãŠã‚ã‚ã‚‹ãã¿ã‚“ãªã«ã§ãã®ã¼ã†ã¨ã‚ˆã°ã‚Œ"
      u"ã»ã‚ã‚‰ã‚Œã‚‚ã›ãšãã«ã‚‚ã•ã‚Œãšãã†ã„ã†ã‚‚ã®ã«ã‚ãŸã—ã¯ãªã‚ŠãŸã„";

const char16_t *const CommonChars::arabic =
      // Some Arabic for variety. "What is Unicode?"
      // http://www.unicode.org/standard/translations/arabic.html
      u"ØªØªØ¹Ø§Ù…Ù„ Ø§Ù„Ø­ÙˆØ§Ø³ÙŠØ¨ Ø¨Ø§Ù„Ø£Ø³Ø§Ù… Ù…Ø¹ Ø§Ù„Ø£Ø±Ù‚Ø§Ù… ÙÙ‚Ø·ØŒ "
      u"Ùˆ ØªØ®Ø²Ù† Ø§Ù„Ø­Ø±ÙˆÙ Ùˆ Ø§Ù„Ù…Ø­Ø§Ø±Ù "
      u"Ø§Ù„Ø£Ø®Ø±Ù‰ Ø¨ØªØ®ØµÙŠØµ Ø±Ù‚Ù… Ù„ÙƒÙ„ ÙˆØ§Ø­Ø¯ "
      u"Ù…Ù†Ù‡Ø§. Ù‚Ø¨Ù„ Ø§Ø®ØªØ±Ø§Ø¹ ÙŠÙˆÙ†ÙŠÙƒÙˆØ¯ ÙƒØ§Ù† Ù‡Ù†Ø§Ùƒ ";

// TODO: class BenchmarkPerCodePoint?

class Operation {
public:
    Operation() {}
    virtual ~Operation();
    virtual double call(int32_t iterations, int32_t pieceLength) = 0;

protected:
    UTimer startTime;
};

Operation::~Operation() {}

const int32_t kLengths[] = { 5, 12, 30, 100, 1000, 10000 };

int32_t getMaxLength() { return kLengths[UPRV_LENGTHOF(kLengths) - 1]; }

// Returns seconds per code point.
double measure(Operation &op, int32_t pieceLength) {
    // Increase the number of iterations until we use at least one second.
    int32_t iterations = 1;
    for (;;) {
        double seconds = op.call(iterations, pieceLength);
        if (seconds >= 1) {
            if (iterations > 1) {
                return seconds / (iterations * pieceLength);
            } else {
                // Run it once more, to avoid measuring only the warm-up.
                return op.call(1, pieceLength) / (iterations * pieceLength);
            }
        }
        if (seconds < 0.01) {
            iterations *= 10;
        } else if (seconds < 0.55) {
            iterations *= 1.1 / seconds;
        } else {
            iterations *= 2;
        }
    }
}

void benchmark(const char *name, Operation &op) {
    for (int32_t i = 0; i < UPRV_LENGTHOF(kLengths); ++i) {
        int32_t pieceLength = kLengths[i];
        double secPerCp = measure(op, pieceLength);
        printf("%s  %6d  %12f ns/cp\n", name, static_cast<int>(pieceLength), secPerCp * 1000000000);
    }
    puts("");
}

class NormalizeUTF16 : public Operation {
public:
    NormalizeUTF16(const Normalizer2 &n2, const UnicodeString &text) :
            norm2(n2), src(text), s(src.getBuffer()) {}
    virtual ~NormalizeUTF16();
    double call(int32_t iterations, int32_t pieceLength) override;

private:
    const Normalizer2 &norm2;
    UnicodeString src;
    const char16_t *s;
    UnicodeString dest;
};

NormalizeUTF16::~NormalizeUTF16() {}

// Assumes all BMP characters.
double NormalizeUTF16::call(int32_t iterations, int32_t pieceLength) {
    int32_t start = 0;
    int32_t limit = src.length() - pieceLength;
    UnicodeString piece;
    UErrorCode errorCode = U_ZERO_ERROR;
    utimer_getTime(&startTime);
    for (int32_t i = 0; i < iterations; ++i) {
        piece.setTo(false, s + start, pieceLength);
        norm2.normalize(piece, dest, errorCode);
        start = (start + pieceLength) % limit;
    }
    return utimer_getElapsedSeconds(&startTime);
}

class NormalizeUTF8 : public Operation {
public:
    NormalizeUTF8(const Normalizer2 &n2, const UnicodeString &text) : norm2(n2), sink(&dest) {
        offsets = CommonChars::toUTF8WithOffsets(text, src, numCodePoints);
        s = src.data();
    }
    virtual ~NormalizeUTF8();
    double call(int32_t iterations, int32_t pieceLength) override;

private:
    const Normalizer2 &norm2;
    std::string src;
    const char *s;
    int32_t *offsets;
    int32_t numCodePoints;
    std::string dest;
    icu::StringByteSink<std::string> sink;
};

NormalizeUTF8::~NormalizeUTF8() {
    delete[] offsets;
}

double NormalizeUTF8::call(int32_t iterations, int32_t pieceLength) {
    int32_t start = 0;
    int32_t limit = numCodePoints - pieceLength;
    UErrorCode errorCode = U_ZERO_ERROR;
    utimer_getTime(&startTime);
    for (int32_t i = 0; i < iterations; ++i) {
        int32_t start8 = offsets[start];
        int32_t limit8 = offsets[start + pieceLength];
        icu::StringPiece piece(s + start8, limit8 - start8);
        norm2.normalizeUTF8(0, piece, sink, nullptr, errorCode);
        start = (start + pieceLength) % limit;
    }
    return utimer_getElapsedSeconds(&startTime);
}

}  // namespace

extern int main(int /*argc*/, const char * /*argv*/[]) {
    // More than the longest piece length so that we read from different parts of the string
    // for that piece length.
    int32_t maxLength = getMaxLength() * 10;
    UErrorCode errorCode = U_ZERO_ERROR;
    const Normalizer2 *nfc = Normalizer2::getNFCInstance(errorCode);
    const Normalizer2 *nfkc_cf = Normalizer2::getNFKCCasefoldInstance(errorCode);
    if (U_FAILURE(errorCode)) {
        fprintf(stderr,
                "simplenormperf: failed to get Normalizer2 instances - %s\n",
                u_errorName(errorCode));
    }
    {
        // Base line: Should remain in the fast loop without trie lookups.
        NormalizeUTF16 op(*nfc, CommonChars::getLatin1(maxLength));
        benchmark("NFC/UTF-16/latin1", op);
    }
    {
        // Base line 2: Read UTF-8, trie lookups, but should have nothing to do.
        NormalizeUTF8 op(*nfc, CommonChars::getJapanese(maxLength));
        benchmark("NFC/UTF-8/japanese", op);
    }
    {
        NormalizeUTF16 op(*nfkc_cf, CommonChars::getMixed(maxLength));
        benchmark("NFKC_CF/UTF-16/mixed", op);
    }
    {
        NormalizeUTF16 op(*nfkc_cf, CommonChars::getLowercaseLatin1(maxLength));
        benchmark("NFKC_CF/UTF-16/lowercaseLatin1", op);
    }
    {
        NormalizeUTF16 op(*nfkc_cf, CommonChars::getJapanese(maxLength));
        benchmark("NFKC_CF/UTF-16/japanese", op);
    }
    {
        NormalizeUTF8 op(*nfkc_cf, CommonChars::getMixed(maxLength));
        benchmark("NFKC_CF/UTF-8/mixed", op);
    }
    {
        NormalizeUTF8 op(*nfkc_cf, CommonChars::getLowercaseLatin1(maxLength));
        benchmark("NFKC_CF/UTF-8/lowercaseLatin1", op);
    }
    {
        NormalizeUTF8 op(*nfkc_cf, CommonChars::getJapanese(maxLength));
        benchmark("NFKC_CF/UTF-8/japanese", op);
    }
    return 0;
}
