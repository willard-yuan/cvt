#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>
#include <utility>
#include <stack>
#include <algorithm>
#include <iterator>
#include <regex>
#include <locale>
#include <codecvt>

namespace cvtk {
class StringUtil {
 public:
  static std::wstring UTF8ToWide(const std::string& source)
  {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
    return conv.from_bytes(source);
  }

  static std::string WideToUTF8(const std::wstring& source)
  {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
    return conv.to_bytes(source);
  }

  static void SplitSentence(const std::string& src, std::vector<std::string>* dst) {
    std::wstring w_src = UTF8ToWide(src);
    int index = 0;
    for (int i = 0; i < w_src.size(); i++) {
      if (symbol.find(w_src[i]) != symbol.end()) {
        if (i == index) {
          index++;
          continue;
        }
        std::wstring w_tmp = w_src.substr(index, i-index);
        std::string tmp = ::base::WideToUTF8(w_tmp);
        index = i + 1;
        dst->push_back(tmp);
      }
    }
    if (index < w_src.size()) {
      std::wstring w_tmp = w_src.substr(index, w_src.size() - index);
      std::string tmp = WideToUTF8(w_tmp);
      dst->push_back(tmp);
    }
  }

  static std::vector<std::string> SegSentence(const std::string& src) {
    std::regex re{"，|？|。|：| |,|!|！|\\...|\\……|\\?|@"};
    return std::vector<std::string> {
        std::sregex_token_iterator(src.begin(), src.end(), re, -1),
        std::sregex_token_iterator()
    };
  }

  static std::string FilterSpecialChars(const std::string& text) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
    std::wregex pattern(L"[^a-zA-Z0-9\u2e80-\u2fd5\u3190-\u319f\u3400-\u4dbf\u4e00-\u9fcc\uf900-\ufaad]");
    return conv.to_bytes(regex_replace(conv.from_bytes(text), pattern,
                std::wstring(L""), std::regex_constants::match_default));
  }

  static void GetNgram(const std::string& src, int N, std::vector<std::pair<std::string, int>> *res) {
    std::wstring w_src, w_tmp;
    std::string tmp = UTF8ToWide(src);
    for (int i = 0; i < (int)w_src.size() - N + 1; i++) {
      w_tmp = w_src.substr(i, N);
      tmp = WideToUTF8(w_tmp);
      res->push_back(std::make_pair(tmp, i));
    }
  }

 private:
  static std::unordered_set<wchar_t> symbol;
};
}  // namespace cvtk
