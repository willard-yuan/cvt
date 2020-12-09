#include "string_util.h"

namespace cvtk {
std::unordered_set<wchar_t> StringUtil::symbol = {L',', L'，', L'.', L'。', L'!', L'！', L'?', L'？'};
}  // namespace cvtk
