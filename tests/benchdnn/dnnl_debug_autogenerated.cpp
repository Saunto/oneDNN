/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// DO NOT EDIT, AUTO-GENERATED
// Use this script to update the file: scripts/generate_dnnl_debug.py

// clang-format off

#include <assert.h>
#include <string.h>

#include "oneapi/dnnl/dnnl_debug.h"

#include "dnnl_debug.hpp"

#include "src/common/z_magic.hpp"

dnnl_data_type_t str2dt(const char *str) {
#define CASE(_case) do { \
    if (!strcmp(STRINGIFY(_case), str) \
            || !strcmp("dnnl_" STRINGIFY(_case), str)) \
        return CONCAT2(dnnl_, _case); \
} while (0)
    CASE(f16);
    CASE(bf16);
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
    CASE(f64);
    CASE(data_type_max);
#undef CASE
    if (!strcmp("undef", str) || !strcmp("dnnl_data_type_undef", str))
        return dnnl_data_type_undef;
    assert(!"unknown dt");
    return dnnl_data_type_undef;
}

dnnl_format_tag_t str2fmt_tag(const char *str) {
#define CASE(_case) do { \
    if (!strcmp(STRINGIFY(_case), str) \
            || !strcmp("dnnl_" STRINGIFY(_case), str)) \
        return CONCAT2(dnnl_, _case); \
} while (0)
    CASE(a);
    CASE(ab);
    CASE(abc);
    CASE(abcd);
    CASE(abcde);
    CASE(abcdef);
    CASE(abcdefg);
    CASE(abcdefgh);
    CASE(abcdefghi);
    CASE(abcdefghij);
    CASE(abcdefghijk);
    CASE(abcdefghijkl);
    CASE(ba);
    CASE(acb);
    CASE(bac);
    CASE(bca);
    CASE(cab);
    CASE(cba);
    CASE(abdc);
    CASE(acbd);
    CASE(acdb);
    CASE(adbc);
    CASE(adcb);
    CASE(bacd);
    CASE(bcda);
    CASE(cdab);
    CASE(cdba);
    CASE(dcab);
    CASE(abced);
    CASE(abdec);
    CASE(acbde);
    CASE(acdeb);
    CASE(adecb);
    CASE(bacde);
    CASE(bcdea);
    CASE(cdeab);
    CASE(cdeba);
    CASE(decab);
    CASE(abcdfe);
    CASE(abdefc);
    CASE(abdfce);
    CASE(acbdef);
    CASE(adefcb);
    CASE(defcab);
    CASE(abcdegf);
    CASE(abcdefhg);
    CASE(abcdefgih);
    CASE(abcdefghji);
    CASE(abcdefghikj);
    CASE(abcdefghijlk);
    CASE(Abc16a);
    CASE(ABc16a16b);
    CASE(ABc32a32b);
    CASE(ABc4a4b);
    CASE(aBc16b);
    CASE(ABc16b16a);
    CASE(Abc4a);
    CASE(aBc32b);
    CASE(aBc4b);
    CASE(ABc4b16a4b);
    CASE(ABc2b8a4b);
    CASE(ABc16b16a4b);
    CASE(ABc16b16a2b);
    CASE(ABc4b4a);
    CASE(ABc8a16b2a);
    CASE(ABc8a8b);
    CASE(ABc8a4b);
    CASE(aBc8b);
    CASE(ABc8b16a2b);
    CASE(BAc8a16b2a);
    CASE(ABc8b8a);
    CASE(Abcd16a);
    CASE(Abcd8a);
    CASE(ABcd16a16b);
    CASE(Abcd32a);
    CASE(ABcd32a32b);
    CASE(aBcd16b);
    CASE(ABcd16b16a);
    CASE(aBCd16b16c);
    CASE(aBCd16c16b);
    CASE(Abcd4a);
    CASE(aBcd32b);
    CASE(aBcd4b);
    CASE(ABcd4b16a4b);
    CASE(ABcd16b16a4b);
    CASE(ABcd16b16a2b);
    CASE(ABcd4b4a);
    CASE(ABcd4a4b);
    CASE(aBCd2c4b2c);
    CASE(aBCd4b8c2b);
    CASE(aBCd4c16b4c);
    CASE(aBCd2c8b4c);
    CASE(aBCd16c16b4c);
    CASE(aBCd16c16b2c);
    CASE(aBCd4c4b);
    CASE(aBCd4b4c);
    CASE(ABcd8a16b2a);
    CASE(ABcd2b8a4b);
    CASE(ABcd8a8b);
    CASE(ABcd8a4b);
    CASE(aBcd8b);
    CASE(aBCd4c8b2c);
    CASE(ABcd8b16a2b);
    CASE(aBCd8b16c2b);
    CASE(BAcd8a16b2a);
    CASE(ABcd8b8a);
    CASE(aBCd8b8c);
    CASE(aBCd8b4c);
    CASE(aBCd8c16b2c);
    CASE(ABcde8a16b2a);
    CASE(aCBd8b16c2b);
    CASE(aBCd8c8b);
    CASE(Abcde16a);
    CASE(Abcde32a);
    CASE(ABcde16a16b);
    CASE(BAcde8a16b2a);
    CASE(aBCd2b4c2b);
    CASE(ABcde4b16a4b);
    CASE(ABcde2b8a4b);
    CASE(aBcde16b);
    CASE(ABcde16b16a);
    CASE(aBCde16b16c);
    CASE(aBCde16c16b);
    CASE(aBCde2c8b4c);
    CASE(Abcde4a);
    CASE(aBcde32b);
    CASE(aBcde4b);
    CASE(ABcde4b4a);
    CASE(ABcde4a4b);
    CASE(aBCde4b4c);
    CASE(aBCde2c4b2c);
    CASE(aBCde4b8c2b);
    CASE(aBCde4c16b4c);
    CASE(aBCde16c16b4c);
    CASE(aBCde16c16b2c);
    CASE(aBCde4c4b);
    CASE(Abcde8a);
    CASE(ABcde8a8b);
    CASE(ABcde8a4b);
    CASE(BAcde16b16a);
    CASE(aBcde8b);
    CASE(ABcde8b16a2b);
    CASE(aBCde8b16c2b);
    CASE(aBCde4c8b2c);
    CASE(aCBde8b16c2b);
    CASE(ABcde8b8a);
    CASE(ABcde32a32b);
    CASE(aBCde8b8c);
    CASE(aBCde8b4c);
    CASE(ABc4a8b8a4b);
    CASE(ABcd4a8b8a4b);
    CASE(ABcde4a8b8a4b);
    CASE(BAc4b8a8b4a);
    CASE(BAcd4b8a8b4a);
    CASE(BAcde4b8a8b4a);
    CASE(ABcd2a8b8a2b);
    CASE(aBCd4b8c8b4c);
    CASE(aBCde4b8c8b4c);
    CASE(aBCde2b8c8b2c);
    CASE(aBCde8c16b2c);
    CASE(aBCde8c8b);
    CASE(aBCde2b4c2b);
    CASE(aBcdef16b);
    CASE(aBCdef16b16c);
    CASE(aBCdef16c16b);
    CASE(aBCdef4c16b4c);
    CASE(aBCdef2c8b4c);
    CASE(aBCdef4c8b2c);
    CASE(aBCdef2b4c2b);
    CASE(aBcdef4b);
    CASE(aBCdef4c4b);
    CASE(aBCdef4b4c);
    CASE(aBCdef2c4b2c);
    CASE(aBCdef4b8c2b);
    CASE(aBCdef8b8c);
    CASE(aBCdef8b4c);
    CASE(aBCdef8c16b2c);
    CASE(aBCdef4b8c8b4c);
    CASE(aBCdef8b16c2b);
    CASE(aCBdef8b16c2b);
    CASE(aBCdef8c8b);
    CASE(aBdc16b);
    CASE(aBdC16b2c);
    CASE(aBdC16b4c);
    CASE(aBdc4b);
    CASE(aBdc8b);
    CASE(aBdec16b);
    CASE(aBdeC16b2c);
    CASE(aBdeC16b4c);
    CASE(aBdec32b);
    CASE(aBdec4b);
    CASE(aBdec8b);
    CASE(aBdefc16b);
    CASE(aBdefC16b2c);
    CASE(aCBdef16c16b);
    CASE(aBdefc4b);
    CASE(aBdefc8b);
    CASE(Abcdef16a);
    CASE(Abcdef32a);
    CASE(aBedc16b);
    CASE(Acb16a);
    CASE(AcB16a2b);
    CASE(AcB16a4b);
    CASE(Acb4a);
    CASE(Acb8a);
    CASE(aCBd16b16c);
    CASE(aCBd16c16b);
    CASE(aCBde16b16c);
    CASE(aCBde16c16b);
    CASE(Acdb16a);
    CASE(AcdB16a2b);
    CASE(AcdB16a4b);
    CASE(Acdb32a);
    CASE(Acdb4a);
    CASE(Acdb8a);
    CASE(Acdeb16a);
    CASE(AcdeB16a2b);
    CASE(Acdeb4a);
    CASE(Acdeb8a);
    CASE(Adcb16a);
    CASE(BAc16a16b);
    CASE(BAc16b16a);
    CASE(BAcd16a16b);
    CASE(BAcd16b16a);
    CASE(aCBd4c8b8c4b);
    CASE(aCBde4c8b8c4b);
    CASE(aCBdef4c8b8c4b);
    CASE(BAcde16a16b);
    CASE(aCBdef16b16c);
    CASE(ABc16b32a);
    CASE(ABc16b64a);
    CASE(ABc4b32a4b);
    CASE(ABc4b64a4b);
    CASE(ABc8b32a2b);
    CASE(ABc8b64a2b);
    CASE(AB16b16a);
    CASE(AB16b32a);
    CASE(AB16b64a);
    CASE(AB8b16a2b);
    CASE(AB8b32a2b);
    CASE(AB8b64a2b);
    CASE(AB4b16a4b);
    CASE(AB4b32a4b);
    CASE(AB4b64a4b);
    CASE(AB16b16a4b);
    CASE(ABcd16b32a);
    CASE(ABcd16b64a);
    CASE(ABcd4b32a4b);
    CASE(ABcd4b64a4b);
    CASE(ABcd8b32a2b);
    CASE(ABcd8b64a2b);
    CASE(ABcde4b32a4b);
    CASE(ABcde4b64a4b);
    CASE(ABcde16b16a4b);
    CASE(ABcde16b16a2b);
    CASE(ABcde16b32a);
    CASE(ABcde16b64a);
    CASE(ABcde8b32a2b);
    CASE(ABcde8b64a2b);
    CASE(aBCdef16c16b4c);
    CASE(aBCdef16c16b2c);
    CASE(AB32a32b8a4b);
    CASE(AB8a4b);
    CASE(AB32a32b8a2b);
    CASE(AB8a2b);
    CASE(abDc32d);
    CASE(abDC32d4c);
    CASE(abdEc32e);
    CASE(abdEC32e2c);
    CASE(abdEC32e4c);
    CASE(aBdefC16b4c);
    CASE(AcdeB16a4b);
    CASE(ABcd16a16b2a);
    CASE(ABc16a16b2a);
    CASE(aBCd16b16c2b);
    CASE(aBCde16b16c2b);
    CASE(Acb32a);
    CASE(AcB32a2b);
    CASE(AcB32a4b);
    CASE(Acb48a);
    CASE(AcB48a2b);
    CASE(AcB48a4b);
    CASE(Acb64a);
    CASE(AcB64a2b);
    CASE(AcB64a4b);
    CASE(cBa2b);
    CASE(cBa4b);
    CASE(aBdc32b);
    CASE(aBdC32b2c);
    CASE(aBdC32b4c);
    CASE(aBdc48b);
    CASE(aBdC48b2c);
    CASE(aBdC48b4c);
    CASE(aBdc64b);
    CASE(aBdC64b2c);
    CASE(aBdC64b4c);
    CASE(adCb2c);
    CASE(adCb4c);
    CASE(AcdB32a2b);
    CASE(AcdB32a4b);
    CASE(Acdb48a);
    CASE(AcdB48a2b);
    CASE(AcdB48a4b);
    CASE(Acdb64a);
    CASE(AcdB64a2b);
    CASE(AcdB64a4b);
    CASE(cdBa2b);
    CASE(cdBa4b);
    CASE(aBdeC32b2c);
    CASE(aBdeC32b4c);
    CASE(aBdec48b);
    CASE(aBdeC48b2c);
    CASE(aBdeC48b4c);
    CASE(aBdec64b);
    CASE(aBdeC64b2c);
    CASE(aBdeC64b4c);
    CASE(adeCb2c);
    CASE(adeCb4c);
    CASE(Acdeb32a);
    CASE(AcdeB32a2b);
    CASE(AcdeB32a4b);
    CASE(Acdeb48a);
    CASE(AcdeB48a2b);
    CASE(AcdeB48a4b);
    CASE(Acdeb64a);
    CASE(AcdeB64a2b);
    CASE(AcdeB64a4b);
    CASE(cdeBa2b);
    CASE(cdeBa4b);
    CASE(aBdefc32b);
    CASE(aBdefC32b2c);
    CASE(aBdefC32b4c);
    CASE(aBdefc48b);
    CASE(aBdefC48b2c);
    CASE(aBdefC48b4c);
    CASE(aBdefc64b);
    CASE(aBdefC64b2c);
    CASE(aBdefC64b4c);
    CASE(adefCb2c);
    CASE(adefCb4c);
    CASE(AB16b32a4b);
    CASE(AB16b48a4b);
    CASE(AB16b64a4b);
    CASE(AB16b16a2b);
    CASE(AB16b32a2b);
    CASE(AB16b48a2b);
    CASE(AB16b64a2b);
    CASE(ABc16b32a4b);
    CASE(ABc16b48a4b);
    CASE(ABc16b64a4b);
    CASE(ABc16b32a2b);
    CASE(ABc16b48a2b);
    CASE(ABc16b64a2b);
    CASE(ABcd16b32a4b);
    CASE(ABcd16b48a4b);
    CASE(ABcd16b64a4b);
    CASE(ABcd16b32a2b);
    CASE(ABcd16b48a2b);
    CASE(ABcd16b64a2b);
    CASE(ABcde16b32a4b);
    CASE(ABcde16b48a4b);
    CASE(ABcde16b64a4b);
    CASE(ABcde16b32a2b);
    CASE(ABcde16b48a2b);
    CASE(ABcde16b64a2b);
    CASE(ABc32a16b);
    CASE(ABcd32a16b);
    CASE(ABcde32a16b);
    CASE(AB48a16b);
    CASE(AB48a32b);
    CASE(ABc40a16b);
    CASE(ABc40a32b);
    CASE(aBC48b16c);
    CASE(aBC48b32c);
    CASE(ABcd40a16b);
    CASE(ABcd40a32b);
    CASE(abCd32c);
    CASE(abdCe32c);
    CASE(abdCE32c2e);
    CASE(BA16a16b2a);
    CASE(BA16a32b2a);
    CASE(BA16a48b2a);
    CASE(BA16a64b2a);
    CASE(BA16a16b4a);
    CASE(BA16a32b4a);
    CASE(BA16a48b4a);
    CASE(BA16a64b4a);
    CASE(ABcd8a2b);
    CASE(aBdeC16c16b2c);
    CASE(aBdeC16c16b4c);
    CASE(aBdefC16c16b2c);
    CASE(AcB16b16a2b);
    CASE(AcB16b16a4b);
    CASE(AcdB16b16a2b);
    CASE(AcdB16b16a4b);
    CASE(AcdeB16b16a2b);
    CASE(aBdefC16c16b4c);
    CASE(AcdeB16b16a4b);
    CASE(AcB16b32a2b);
    CASE(AcB16b32a4b);
    CASE(AcB16b48a2b);
    CASE(AcB16b48a4b);
    CASE(AcB16b64a2b);
    CASE(AcB16b64a4b);
    CASE(aBdC16c16b2c);
    CASE(aBdC16c16b4c);
    CASE(aBdC16c32b2c);
    CASE(aBdC16c32b4c);
    CASE(aBdC16c48b2c);
    CASE(aBdC16c48b4c);
    CASE(aBdC16c64b2c);
    CASE(aBdC16c64b4c);
    CASE(AcdB16b32a2b);
    CASE(AcdB16b32a4b);
    CASE(AcdB16b48a2b);
    CASE(AcdB16b48a4b);
    CASE(AcdB16b64a2b);
    CASE(AcdB16b64a4b);
    CASE(aBdeC16c32b2c);
    CASE(aBdeC16c32b4c);
    CASE(aBdeC16c48b2c);
    CASE(aBdeC16c48b4c);
    CASE(aBdeC16c64b2c);
    CASE(aBdeC16c64b4c);
    CASE(AcdeB16b32a2b);
    CASE(AcdeB16b32a4b);
    CASE(AcdeB16b48a2b);
    CASE(AcdeB16b48a4b);
    CASE(AcdeB16b64a2b);
    CASE(AcdeB16b64a4b);
    CASE(aBdefC16c32b2c);
    CASE(aBdefC16c32b4c);
    CASE(aBdefC16c48b2c);
    CASE(aBdefC16c48b4c);
    CASE(aBdefC16c64b2c);
    CASE(aBdefC16c64b4c);
    CASE(decbA16a);
    CASE(ABc4a2b);
    CASE(ABc8a2b);
    CASE(aBCd8b2c);
    CASE(ABcde4a2b);
    CASE(ABcde8a2b);
    CASE(ABcde40a16b);
    CASE(ABcde40a32b);
    CASE(aBCde8b2c);
    CASE(ABcde4a8b8a2b);
    CASE(ABcd4a8b8a2b);
    CASE(ABc4a8b8a2b);
    CASE(aBCdef4b8c8b2c);
    CASE(aBCde4b8c8b2c);
    CASE(aBCd4b8c8b2c);
    CASE(BAcde4b8a8b2a);
    CASE(BAcd4b8a8b2a);
    CASE(BAc4b8a8b2a);
    CASE(aCBdef4c8b8c2b);
    CASE(aCBde4c8b8c2b);
    CASE(aCBd4c8b8c2b);
    CASE(aBCdef8b2c);
    CASE(AB32a16b);
    CASE(AB32a32b);
    CASE(BA4b8a8b2a);
    CASE(BA4b8a8b4a);
    CASE(aBC32b16c);
    CASE(aBC32b32c);
    CASE(aCB4c8b8c2b);
    CASE(aCB4c8b8c4b);
    CASE(ABcd4a2b);
    CASE(ABc2b8a16b4a);
    CASE(ABcd2b8a16b4a);
    CASE(ABcde2b8a16b4a);
    CASE(ABc2a8b16a4b);
    CASE(ABc2a8b16a2b);
    CASE(ABc2b32a8b);
    CASE(ABcd2a8b16a4b);
    CASE(ABcd2a8b16a2b);
    CASE(aCBd2c8b16c2b);
    CASE(ABcd2b32a8b);
    CASE(aBCd2c8b16c2b);
    CASE(ABcde2a8b16a4b);
    CASE(ABcde2a8b16a2b);
    CASE(aCBde2c8b16c2b);
    CASE(ABcde2b32a8b);
    CASE(aBC2b8c16b2c);
    CASE(aBCd2b8c16b2c);
    CASE(aBCde2b8c16b2c);
    CASE(aBCdef2b8c16b2c);
    CASE(BAcde2b8a16b4a);
    CASE(BAcd2b8a16b4a);
    CASE(BAc2b8a16b4a);
    CASE(BAcde2b8a16b2a);
    CASE(BAcd2b8a16b2a);
    CASE(BAc2b8a16b2a);
    CASE(aBCde2c8b16c2b);
    CASE(aBCdef2c8b16c2b);
    CASE(aCBdef2c8b16c2b);
    CASE(aBCd2b8c16b4c);
    CASE(aBCde2b8c16b4c);
    CASE(BA4b8a16b2a);
    CASE(BA4b8a16b4a);
    CASE(aCB4c8b16c2b);
    CASE(aCB4c8b16c4b);
    CASE(BA16a16b);
    CASE(BA16a32b);
    CASE(BA16a48b);
    CASE(BA16a64b);
    CASE(aCB16c2b);
    CASE(aCB16c4b);
    CASE(BA16b2a);
    CASE(BA16b4a);
    CASE(aBC16b16c);
    CASE(aBC16b32c);
    CASE(AB16a16b);
    CASE(AB16a32b);
    CASE(ABcde16a16b2a);
    CASE(aBCdef16b16c2b);
    CASE(Acedb16a);
    CASE(aBdfec16b);
    CASE(abdEC64e2c);
    CASE(abdEC64e4c);
    CASE(aCB16b16c);
    CASE(aCB16b32c);
    CASE(aCB16b48c);
    CASE(aCB16b64c);
    CASE(aCB16b16c2b);
    CASE(aCB16b32c2b);
    CASE(aCB16b48c2b);
    CASE(aCB16b64c2b);
    CASE(aCB16b16c4b);
    CASE(aCB16b32c4b);
    CASE(aCB16b48c4b);
    CASE(aCB16b64c4b);
    CASE(abCd4c);
    CASE(abCde4c);
    CASE(abCdef4c);
    CASE(abCde32c);
    CASE(abCdef32c);
    CASE(ABcd16a32b);
    CASE(decbA8a);
    CASE(aCdefB16b32c2b);
    CASE(aCdefB16b32c4b);
    CASE(aCdefB16b48c2b);
    CASE(aCdefB16b48c4b);
    CASE(aCdefB16b64c2b);
    CASE(aCdefB16b64c4b);
    CASE(BcdeA16a32b2a);
    CASE(BcdeA16a32b4a);
    CASE(BcdeA16a48b2a);
    CASE(BcdeA16a48b4a);
    CASE(BcdeA16a64b2a);
    CASE(BcdeA16a64b4a);
    CASE(aCdefb32c);
    CASE(aCdefB32c2b);
    CASE(aCdefB32c4b);
    CASE(aCdefb48c);
    CASE(aCdefB48c2b);
    CASE(aCdefB48c4b);
    CASE(aCdefb64c);
    CASE(aCdefB64c2b);
    CASE(aCdefB64c4b);
    CASE(Bcdea32b);
    CASE(BcdeA32b2a);
    CASE(BcdeA32b4a);
    CASE(Bcdea48b);
    CASE(BcdeA48b2a);
    CASE(BcdeA48b4a);
    CASE(Bcdea64b);
    CASE(BcdeA64b2a);
    CASE(BcdeA64b4a);
    CASE(Bca32b);
    CASE(BcA32b2a);
    CASE(BcA32b4a);
    CASE(Bca48b);
    CASE(BcA48b2a);
    CASE(BcA48b4a);
    CASE(Bca64b);
    CASE(BcA64b2a);
    CASE(BcA64b4a);
    CASE(aCdb32c);
    CASE(aCdB32c2b);
    CASE(aCdB32c4b);
    CASE(aCdb48c);
    CASE(aCdB48c2b);
    CASE(aCdB48c4b);
    CASE(aCdb64c);
    CASE(aCdB64c2b);
    CASE(aCdB64c4b);
    CASE(BcA16a16b2a);
    CASE(BcA16a16b4a);
    CASE(BcdA16a16b2a);
    CASE(BcdA16a16b4a);
    CASE(BcdeA16a16b2a);
    CASE(BcdeA16a16b4a);
    CASE(aCdB16b16c2b);
    CASE(aCdB16b16c4b);
    CASE(aCdeB16b16c2b);
    CASE(aCdeB16b16c4b);
    CASE(aCdefB16b16c2b);
    CASE(aCdefB16b16c4b);
    CASE(BcA16a32b2a);
    CASE(BcA16a32b4a);
    CASE(BcA16a48b2a);
    CASE(BcA16a48b4a);
    CASE(BcA16a64b2a);
    CASE(BcA16a64b4a);
    CASE(aCdB16b32c2b);
    CASE(aCdB16b32c4b);
    CASE(aCdB16b48c2b);
    CASE(aCdB16b48c4b);
    CASE(aCdB16b64c2b);
    CASE(aCdB16b64c4b);
    CASE(BcdA16a32b2a);
    CASE(BcdA16a32b4a);
    CASE(BcdA16a48b2a);
    CASE(BcdA16a48b4a);
    CASE(BcdA16a64b2a);
    CASE(BcdA16a64b4a);
    CASE(aCdeB16b32c2b);
    CASE(aCdeB16b32c4b);
    CASE(aCdeB16b48c2b);
    CASE(aCdeB16b48c4b);
    CASE(aCdeB16b64c2b);
    CASE(aCdeB16b64c4b);
    CASE(Bca16b);
    CASE(BcA16b2a);
    CASE(BcA16b4a);
    CASE(Bcda16b);
    CASE(BcdA16b2a);
    CASE(BcdA16b4a);
    CASE(Bcdea16b);
    CASE(BcdeA16b2a);
    CASE(BcdeA16b4a);
    CASE(aCdb16c);
    CASE(aCdB16c2b);
    CASE(aCdB16c4b);
    CASE(aCdeb16c);
    CASE(aCdeB16c2b);
    CASE(aCdeB16c4b);
    CASE(aCdefb16c);
    CASE(aCdefB16c2b);
    CASE(aCdefB16c4b);
    CASE(Bcda32b);
    CASE(BcdA32b2a);
    CASE(BcdA32b4a);
    CASE(Bcda48b);
    CASE(BcdA48b2a);
    CASE(BcdA48b4a);
    CASE(Bcda64b);
    CASE(BcdA64b2a);
    CASE(BcdA64b4a);
    CASE(aCdeb32c);
    CASE(aCdeB32c2b);
    CASE(aCdeB32c4b);
    CASE(aCdeb48c);
    CASE(aCdeB48c2b);
    CASE(aCdeB48c4b);
    CASE(aCdeb64c);
    CASE(aCdeB64c2b);
    CASE(aCdeB64c4b);
    CASE(x);
    CASE(nc);
    CASE(cn);
    CASE(tn);
    CASE(nt);
    CASE(ncw);
    CASE(nwc);
    CASE(nchw);
    CASE(nhwc);
    CASE(chwn);
    CASE(ncdhw);
    CASE(ndhwc);
    CASE(oi);
    CASE(io);
    CASE(oiw);
    CASE(owi);
    CASE(wio);
    CASE(woi);
    CASE(iwo);
    CASE(oihw);
    CASE(hwio);
    CASE(hwoi);
    CASE(ohwi);
    CASE(ihwo);
    CASE(iohw);
    CASE(oidhw);
    CASE(iodhw);
    CASE(dhwio);
    CASE(dhwoi);
    CASE(odhwi);
    CASE(idhwo);
    CASE(goiw);
    CASE(gowi);
    CASE(wigo);
    CASE(goihw);
    CASE(gohwi);
    CASE(hwigo);
    CASE(giohw);
    CASE(goidhw);
    CASE(godhwi);
    CASE(giodhw);
    CASE(dhwigo);
    CASE(tnc);
    CASE(ntc);
    CASE(ldnc);
    CASE(ldigo);
    CASE(ldgoi);
    CASE(ldio);
    CASE(ldoi);
    CASE(ldgo);
    CASE(ldOi32o);
    CASE(ldOI32o4i);
    CASE(ldIo32i);
    CASE(ldgOi32o);
    CASE(ldgOI32o2i);
    CASE(ldgOI32o4i);
    CASE(ldgOI64o2i);
    CASE(ldgOI64o4i);
    CASE(ldgIo32i);
    CASE(ldgIO32i2o);
    CASE(nCdhw32c);
    CASE(nCdhw16c);
    CASE(nCdhw4c);
    CASE(nCdhw8c);
    CASE(nChw32c);
    CASE(nChw16c);
    CASE(nChw4c);
    CASE(nChw8c);
    CASE(nCw32c);
    CASE(nCw16c);
    CASE(nCw4c);
    CASE(nCw8c);
    CASE(NCw16n16c);
    CASE(NCdhw16n16c);
    CASE(NChw16n16c);
    CASE(NCw32n16c);
    CASE(NChw32n16c);
    CASE(NChw16n32c);
    CASE(NCdhw32n16c);
    CASE(NCw32n32c);
    CASE(NChw32n32c);
    CASE(NCdhw32n32c);
    CASE(OI16i16o);
    CASE(OI16i32o);
    CASE(OI16i64o);
    CASE(OI8i16o2i);
    CASE(OI8i32o2i);
    CASE(OI8i64o2i);
    CASE(OI4i16o4i);
    CASE(OI4i32o4i);
    CASE(OI4i64o4i);
    CASE(OI16i16o4i);
    CASE(IOw16o16i);
    CASE(IOw16i16o);
    CASE(OIw16i16o);
    CASE(OIw16i32o);
    CASE(OIw16i64o);
    CASE(OIw16o16i);
    CASE(Oiw16o);
    CASE(OIw4i16o4i);
    CASE(OIw4i32o4i);
    CASE(OIw4i64o4i);
    CASE(OIw2i8o4i);
    CASE(OIw16i16o4i);
    CASE(OIw16i16o2i);
    CASE(OIw16o16i2o);
    CASE(OIw4i4o);
    CASE(OIw4o4i);
    CASE(Oiw4o);
    CASE(OIw8i16o2i);
    CASE(OIw8i32o2i);
    CASE(OIw8i64o2i);
    CASE(OIw8i8o);
    CASE(OIw8o16i2o);
    CASE(IOw8o16i2o);
    CASE(OIw8o8i);
    CASE(OIw8o4i);
    CASE(Owi16o);
    CASE(OwI16o2i);
    CASE(OwI16o4i);
    CASE(Iwo16i);
    CASE(IwO16i2o);
    CASE(IwO16i4o);
    CASE(Owi4o);
    CASE(Owi8o);
    CASE(IOhw16i16o);
    CASE(IOhw16o16i);
    CASE(Ohwi16o);
    CASE(OhwI16o2i);
    CASE(OhwI16o4i);
    CASE(Ihwo16i);
    CASE(IhwO16i2o);
    CASE(IhwO16i4o);
    CASE(Ohwi32o);
    CASE(Ohwi4o);
    CASE(Ohwi8o);
    CASE(OIhw16i16o);
    CASE(OIhw16i32o);
    CASE(OIhw16i64o);
    CASE(OIhw16o16i);
    CASE(Oihw16o);
    CASE(OIhw4i16o4i);
    CASE(OIhw4i32o4i);
    CASE(OIhw4i64o4i);
    CASE(OIhw16i16o4i);
    CASE(OIhw16i16o2i);
    CASE(OIhw16o16i2o);
    CASE(OIhw4i4o);
    CASE(OIhw4o4i);
    CASE(Oihw4o);
    CASE(OIhw8i16o2i);
    CASE(OIhw8i32o2i);
    CASE(OIhw8i64o2i);
    CASE(OIhw8i8o);
    CASE(OIhw8o16i2o);
    CASE(OIhw2i8o4i);
    CASE(IOhw8o16i2o);
    CASE(OIhw8o8i);
    CASE(OIhw8o4i);
    CASE(Owhi16o);
    CASE(Odhwi16o);
    CASE(OdhwI16o2i);
    CASE(OdhwI16o4i);
    CASE(Idhwo16i);
    CASE(IdhwO16i2o);
    CASE(IdhwO16i4o);
    CASE(Odhwi4o);
    CASE(Odhwi8o);
    CASE(Odwhi16o);
    CASE(OIdhw16i16o);
    CASE(OIdhw16i32o);
    CASE(OIdhw16i64o);
    CASE(OIdhw16o16i);
    CASE(Oidhw16o);
    CASE(OIdhw4i4o);
    CASE(OIdhw4o4i);
    CASE(Oidhw4o);
    CASE(OIdhw8i16o2i);
    CASE(OIdhw8i32o2i);
    CASE(OIdhw8i64o2i);
    CASE(OIdhw8i8o);
    CASE(OIdhw8o16i2o);
    CASE(IOdhw8o16i2o);
    CASE(OIdhw4i16o4i);
    CASE(OIdhw4i32o4i);
    CASE(OIdhw4i64o4i);
    CASE(OIdhw16i16o4i);
    CASE(OIdhw16i16o2i);
    CASE(OIdhw2i8o4i);
    CASE(OIdhw8o8i);
    CASE(OIdhw8o4i);
    CASE(IOdhw16i16o);
    CASE(OIdhw4o8i8o4i);
    CASE(IOdhw16o16i);
    CASE(OIdhw16o16i2o);
    CASE(Goiw16g);
    CASE(Goiw8g);
    CASE(Goiw4g);
    CASE(gIOw16o16i);
    CASE(gIOw16i16o);
    CASE(gOIw16i16o);
    CASE(gOIw16o16i);
    CASE(gOiw16o);
    CASE(gOIw4i16o4i);
    CASE(gOIw2i8o4i);
    CASE(gOIw16i16o4i);
    CASE(gOIw16i16o2i);
    CASE(gOIw16o16i2o);
    CASE(gOIw4i4o);
    CASE(gOIw4o4i);
    CASE(gOiw4o);
    CASE(gOIw8i16o2i);
    CASE(gOIw8i8o);
    CASE(gOIw8o16i2o);
    CASE(gIOw8o16i2o);
    CASE(gOIw8o8i);
    CASE(gOIw8o4i);
    CASE(gOwi16o);
    CASE(gOwI16o2i);
    CASE(gOwI16o4i);
    CASE(gIwo16i);
    CASE(gIwO16i2o);
    CASE(gIwO16i4o);
    CASE(gOwi4o);
    CASE(gOwi8o);
    CASE(Goiw32g);
    CASE(gOIw2i4o2i);
    CASE(gOIw2o4i2o);
    CASE(gOIw4i8o2i);
    CASE(gOIw4o8i2o);
    CASE(goIw4i);
    CASE(goIw32i);
    CASE(gIOhw16i16o);
    CASE(gIOhw16o16i);
    CASE(gOhwi16o);
    CASE(gOhwI16o2i);
    CASE(gOhwI16o4i);
    CASE(gIhwo16i);
    CASE(gIhwO16i2o);
    CASE(gIhwO16i4o);
    CASE(gOhwi32o);
    CASE(gOhwi4o);
    CASE(gOhwi8o);
    CASE(Goihw16g);
    CASE(gOIhw16i16o);
    CASE(gOIhw16o16i);
    CASE(gOihw16o);
    CASE(gOIhw2i8o4i);
    CASE(gOIhw4i16o4i);
    CASE(gOIhw16i16o4i);
    CASE(gOIhw16i16o2i);
    CASE(gOIhw16o16i2o);
    CASE(gOIhw4i4o);
    CASE(gOIhw4o4i);
    CASE(gOihw4o);
    CASE(Goihw8g);
    CASE(Goihw4g);
    CASE(gOIhw8i16o2i);
    CASE(gOIhw8i8o);
    CASE(gOIhw8o16i2o);
    CASE(gIOhw8o16i2o);
    CASE(gOIhw8o8i);
    CASE(gOIhw8o4i);
    CASE(Goihw32g);
    CASE(gOwhi16o);
    CASE(goIhw4i);
    CASE(goIhw32i);
    CASE(OIw4o8i8o4i);
    CASE(OIhw4o8i8o4i);
    CASE(IOw4i8o8i4o);
    CASE(IOhw4i8o8i4o);
    CASE(IOdhw4i8o8i4o);
    CASE(OIhw2o8i8o2i);
    CASE(gOIw4o8i8o4i);
    CASE(gOIhw4o8i8o4i);
    CASE(gOIdhw4o8i8o4i);
    CASE(gIOw4i8o8i4o);
    CASE(gIOhw4i8o8i4o);
    CASE(gIOdhw4i8o8i4o);
    CASE(gOIhw2o8i8o2i);
    CASE(gOIhw2i4o2i);
    CASE(gOIhw2o4i2o);
    CASE(gOIhw4i8o2i);
    CASE(gOIhw4o8i2o);
    CASE(gIOdhw16i16o);
    CASE(gIOdhw16o16i);
    CASE(gOdhwi16o);
    CASE(gOdhwI16o2i);
    CASE(gOdhwI16o4i);
    CASE(gIdhwo16i);
    CASE(gIdhwO16i2o);
    CASE(gIdhwO16i4o);
    CASE(gOdhwi4o);
    CASE(gOdhwi8o);
    CASE(gOdwhi16o);
    CASE(gOIdhw16i16o);
    CASE(gOIdhw4i16o4i);
    CASE(gOIdhw16i16o4i);
    CASE(gOIdhw2i8o4i);
    CASE(gOIdhw16i16o2i);
    CASE(gOIdhw16o16i);
    CASE(gOIdhw16o16i2o);
    CASE(gOidhw16o);
    CASE(gOIdhw4i4o);
    CASE(gOIdhw4o4i);
    CASE(gOidhw4o);
    CASE(gOIdhw8i16o2i);
    CASE(gOIdhw8i8o);
    CASE(gOIdhw8o16i2o);
    CASE(gIOdhw8o16i2o);
    CASE(gOIdhw8o8i);
    CASE(gOIdhw8o4i);
    CASE(Goidhw16g);
    CASE(Goidhw32g);
    CASE(gOIdhw2i4o2i);
    CASE(gOIdhw4i8o2i);
    CASE(gOIdhw2o4i2o);
    CASE(gOIdhw4o8i2o);
    CASE(goIdhw4i);
    CASE(goIdhw32i);
    CASE(Owi32o);
    CASE(OwI32o2i);
    CASE(OwI32o4i);
    CASE(Owi48o);
    CASE(OwI48o2i);
    CASE(OwI48o4i);
    CASE(Owi64o);
    CASE(OwI64o2i);
    CASE(OwI64o4i);
    CASE(Iwo32i);
    CASE(IwO32i2o);
    CASE(IwO32i4o);
    CASE(Iwo48i);
    CASE(IwO48i2o);
    CASE(IwO48i4o);
    CASE(Iwo64i);
    CASE(IwO64i2o);
    CASE(IwO64i4o);
    CASE(wIo2i);
    CASE(wIo4i);
    CASE(gOwi32o);
    CASE(gOwI32o2i);
    CASE(gOwI32o4i);
    CASE(gOwi48o);
    CASE(gOwI48o2i);
    CASE(gOwI48o4i);
    CASE(gOwi64o);
    CASE(gOwI64o2i);
    CASE(gOwI64o4i);
    CASE(gIwo32i);
    CASE(gIwO32i2o);
    CASE(gIwO32i4o);
    CASE(gIwo48i);
    CASE(gIwO48i2o);
    CASE(gIwO48i4o);
    CASE(gIwo64i);
    CASE(gIwO64i2o);
    CASE(gIwO64i4o);
    CASE(gwio);
    CASE(gwIo2i);
    CASE(gwIo4i);
    CASE(OhwI32o);
    CASE(OhwI32o2i);
    CASE(OhwI32o4i);
    CASE(Ohwi48o);
    CASE(OhwI48o2i);
    CASE(OhwI48o4i);
    CASE(Ohwi64o);
    CASE(OhwI64o2i);
    CASE(OhwI64o4i);
    CASE(Ihwo32i);
    CASE(IhwO32i2o);
    CASE(IhwO32i4o);
    CASE(Ihwo48i);
    CASE(IhwO48i2o);
    CASE(IhwO48i4o);
    CASE(Ihwo64i);
    CASE(IhwO64i2o);
    CASE(IhwO64i4o);
    CASE(hwIo2i);
    CASE(hwIo4i);
    CASE(gOhwI32o);
    CASE(gOhwI32o2i);
    CASE(gOhwI32o4i);
    CASE(gOhwi48o);
    CASE(gOhwI48o2i);
    CASE(gOhwI48o4i);
    CASE(gOhwi64o);
    CASE(gOhwI64o2i);
    CASE(gOhwI64o4i);
    CASE(gIhwo32i);
    CASE(gIhwO32i2o);
    CASE(gIhwO32i4o);
    CASE(gIhwo48i);
    CASE(gIhwO48i2o);
    CASE(gIhwO48i4o);
    CASE(gIhwo64i);
    CASE(gIhwO64i2o);
    CASE(gIhwO64i4o);
    CASE(ghwio);
    CASE(ghwIo2i);
    CASE(ghwIo4i);
    CASE(Odhwi32o);
    CASE(OdhwI32o2i);
    CASE(OdhwI32o4i);
    CASE(Odhwi48o);
    CASE(OdhwI48o2i);
    CASE(OdhwI48o4i);
    CASE(Odhwi64o);
    CASE(OdhwI64o2i);
    CASE(OdhwI64o4i);
    CASE(Idhwo32i);
    CASE(IdhwO32i2o);
    CASE(IdhwO32i4o);
    CASE(Idhwo48i);
    CASE(IdhwO48i2o);
    CASE(IdhwO48i4o);
    CASE(Idhwo64i);
    CASE(IdhwO64i2o);
    CASE(IdhwO64i4o);
    CASE(dhwIo2i);
    CASE(dhwIo4i);
    CASE(gOdhwi32o);
    CASE(gOdhwI32o2i);
    CASE(gOdhwI32o4i);
    CASE(gOdhwi48o);
    CASE(gOdhwI48o2i);
    CASE(gOdhwI48o4i);
    CASE(gOdhwi64o);
    CASE(gOdhwI64o2i);
    CASE(gOdhwI64o4i);
    CASE(gIdhwo32i);
    CASE(gIdhwO32i2o);
    CASE(gIdhwO32i4o);
    CASE(gIdhwo48i);
    CASE(gIdhwO48i2o);
    CASE(gIdhwO48i4o);
    CASE(gIdhwo64i);
    CASE(gIdhwO64i2o);
    CASE(gIdhwO64i4o);
    CASE(gdhwio);
    CASE(gdhwIo2i);
    CASE(gdhwIo4i);
    CASE(OI16i32o4i);
    CASE(OI16i48o4i);
    CASE(OI16i64o4i);
    CASE(OI16i16o2i);
    CASE(OI16i32o2i);
    CASE(OI16i48o2i);
    CASE(OI16i64o2i);
    CASE(OIw16i32o4i);
    CASE(OIw16i48o4i);
    CASE(OIw16i64o4i);
    CASE(OIw16i32o2i);
    CASE(OIw16i48o2i);
    CASE(OIw16i64o2i);
    CASE(OIhw16i32o4i);
    CASE(OIhw16i48o4i);
    CASE(OIhw16i64o4i);
    CASE(OIhw16i32o2i);
    CASE(OIhw16i48o2i);
    CASE(OIhw16i64o2i);
    CASE(OIdhw16i32o4i);
    CASE(OIdhw16i48o4i);
    CASE(OIdhw16i64o4i);
    CASE(OIdhw16i32o2i);
    CASE(OIdhw16i48o2i);
    CASE(OIdhw16i64o2i);
    CASE(OwI16i16o2i);
    CASE(OwI16i16o4i);
    CASE(OhwI16i16o2i);
    CASE(OhwI16i16o4i);
    CASE(OdhwI16i16o2i);
    CASE(OdhwI16i16o4i);
    CASE(IwO16o16i2o);
    CASE(IwO16o16i4o);
    CASE(IhwO16o16i2o);
    CASE(IhwO16o16i4o);
    CASE(IdhwO16o16i2o);
    CASE(IdhwO16o16i4o);
    CASE(gOwI16i16o2i);
    CASE(gOwI16i16o4i);
    CASE(gOhwI16i16o2i);
    CASE(gOhwI16i16o4i);
    CASE(gOdhwI16i16o2i);
    CASE(gOdhwI16i16o4i);
    CASE(gIwO16o16i2o);
    CASE(gIwO16o16i4o);
    CASE(gIhwO16o16i2o);
    CASE(gIhwO16o16i4o);
    CASE(gIdhwO16o16i2o);
    CASE(gIdhwO16o16i4o);
    CASE(OwI16i32o2i);
    CASE(OwI16i32o4i);
    CASE(OwI16i48o2i);
    CASE(OwI16i48o4i);
    CASE(OwI16i64o2i);
    CASE(OwI16i64o4i);
    CASE(IwO16o32i2o);
    CASE(IwO16o32i4o);
    CASE(IwO16o48i2o);
    CASE(IwO16o48i4o);
    CASE(IwO16o64i2o);
    CASE(IwO16o64i4o);
    CASE(gOwI16i32o2i);
    CASE(gOwI16i32o4i);
    CASE(gOwI16i48o2i);
    CASE(gOwI16i48o4i);
    CASE(gOwI16i64o2i);
    CASE(gOwI16i64o4i);
    CASE(gIwO16o32i2o);
    CASE(gIwO16o32i4o);
    CASE(gIwO16o48i2o);
    CASE(gIwO16o48i4o);
    CASE(gIwO16o64i2o);
    CASE(gIwO16o64i4o);
    CASE(OhwI16i32o2i);
    CASE(OhwI16i32o4i);
    CASE(OhwI16i48o2i);
    CASE(OhwI16i48o4i);
    CASE(OhwI16i64o2i);
    CASE(OhwI16i64o4i);
    CASE(IhwO16o32i2o);
    CASE(IhwO16o32i4o);
    CASE(IhwO16o48i2o);
    CASE(IhwO16o48i4o);
    CASE(IhwO16o64i2o);
    CASE(IhwO16o64i4o);
    CASE(gOhwI16i32o2i);
    CASE(gOhwI16i32o4i);
    CASE(gOhwI16i48o2i);
    CASE(gOhwI16i48o4i);
    CASE(gOhwI16i64o2i);
    CASE(gOhwI16i64o4i);
    CASE(gIhwO16o32i2o);
    CASE(gIhwO16o32i4o);
    CASE(gIhwO16o48i2o);
    CASE(gIhwO16o48i4o);
    CASE(gIhwO16o64i2o);
    CASE(gIhwO16o64i4o);
    CASE(OdhwI16i32o2i);
    CASE(OdhwI16i32o4i);
    CASE(OdhwI16i48o2i);
    CASE(OdhwI16i48o4i);
    CASE(OdhwI16i64o2i);
    CASE(OdhwI16i64o4i);
    CASE(IdhwO16o32i2o);
    CASE(IdhwO16o32i4o);
    CASE(IdhwO16o48i2o);
    CASE(IdhwO16o48i4o);
    CASE(IdhwO16o64i2o);
    CASE(IdhwO16o64i4o);
    CASE(gOdhwI16i32o2i);
    CASE(gOdhwI16i32o4i);
    CASE(gOdhwI16i48o2i);
    CASE(gOdhwI16i48o4i);
    CASE(gOdhwI16i64o2i);
    CASE(gOdhwI16i64o4i);
    CASE(gIdhwO16o32i2o);
    CASE(gIdhwO16o32i4o);
    CASE(gIdhwO16o48i2o);
    CASE(gIdhwO16o48i4o);
    CASE(gIdhwO16o64i2o);
    CASE(gIdhwO16o64i4o);
    CASE(hwioG16g);
    CASE(hwioG8g);
    CASE(NCdhw40n16c);
    CASE(NCw40n16c);
    CASE(NChw40n16c);
    CASE(NCw40n32c);
    CASE(NChw40n32c);
    CASE(NCdhw40n32c);
    CASE(OIdhw4o8i8o2i);
    CASE(OIhw4o8i8o2i);
    CASE(OIw4o8i8o2i);
    CASE(gOIdhw4o8i8o2i);
    CASE(gOIhw4o8i8o2i);
    CASE(gOIw4o8i8o2i);
    CASE(IOdhw4i8o8i2o);
    CASE(IOhw4i8o8i2o);
    CASE(IOw4i8o8i2o);
    CASE(gIOdhw4i8o8i2o);
    CASE(gIOhw4i8o8i2o);
    CASE(gIOw4i8o8i2o);
    CASE(NCw2c32n8c);
    CASE(NChw2c32n8c);
    CASE(NCdhw2c32n8c);
    CASE(OIw2i8o16i4o);
    CASE(OIhw2i8o16i4o);
    CASE(OIdhw2i8o16i4o);
    CASE(OIw2o8i16o4i);
    CASE(OIw2o8i16o2i);
    CASE(IOw2i8o16i4o);
    CASE(IOw2i8o16i2o);
    CASE(OIhw2o8i16o4i);
    CASE(OIhw2o8i16o2i);
    CASE(IOhw2i8o16i4o);
    CASE(IOhw2i8o16i2o);
    CASE(OIdhw2o8i16o4i);
    CASE(OIdhw2o8i16o2i);
    CASE(IOdhw2i8o16i4o);
    CASE(IOdhw2i8o16i2o);
    CASE(gOIw2o8i16o2i);
    CASE(gIOw2i8o16i2o);
    CASE(gIOhw2i8o16i2o);
    CASE(gIOdhw2i8o16i2o);
    CASE(gOIhw2o8i16o2i);
    CASE(gOIdhw2o8i16o2i);
    CASE(gOIw2o8i16o4i);
    CASE(gOIhw2o8i16o4i);
#undef CASE
    if (!strcmp("undef", str) || !strcmp("dnnl_format_tag_undef", str))
        return dnnl_format_tag_undef;
    if (!strcmp("any", str) || !strcmp("dnnl_format_tag_any", str))
        return dnnl_format_tag_any;
    return dnnl_format_tag_last;
}

const char *status2str(dnnl_status_t status) {
    return dnnl_status2str(status);
}

const char *dt2str(dnnl_data_type_t dt) {
    return dnnl_dt2str(dt);
}

const char *fmt_tag2str(dnnl_format_tag_t tag) {
    return dnnl_fmt_tag2str(tag);
}

const char *engine_kind2str(dnnl_engine_kind_t kind) {
    return dnnl_engine_kind2str(kind);
}

const char *scratchpad_mode2str(dnnl_scratchpad_mode_t mode) {
    return dnnl_scratchpad_mode2str(mode);
}

const char *fpmath_mode2str(dnnl_fpmath_mode_t mode) {
    return dnnl_fpmath_mode2str(mode);
}

