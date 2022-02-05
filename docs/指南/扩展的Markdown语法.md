# 扩展的 Markdown 语法

## 内容强调

### 重要内容

适合重要的提示信息

```markdown
!> 重要重要
```

!> 重要重要

### 提示内容

```markdown
?> 提示
```

?> 提示

## 链接

### 忽略编译链接

```markdown
[link](/demo/ ":ignore")
```

[link](/demo/ ":ignore title")

### 设置链接的 target 属性

```markdown
[link](/demo ":target=_blank")
[link](/demo2 ":target=_self")
```

[百度一](https://www.baidu.com ":target=_blank")
[百度二](https://www.baidu.com ":target=_self")

### 禁用链接

[link](/demo ":disabled")

### Github 任务列表

```markdown
- [ ] foo
- bar
- [x] baz
- [] bam <~ not working
  - [ ] bim
  - [ ] lim
```

- [ ] foo
- bar
- [x] baz
- [] bam <~ not working
  - [ ] bim
  - [ ] lim

## 图片

### 设置图片大小

```markdown
![logo](https://docsify.js.org/_media/icon.svg ":size=WIDTHxHEIGHT")
![logo](https://docsify.js.org/_media/icon.svg ":size=50x100")
![logo](https://docsify.js.org/_media/icon.svg ":size=100")

<!-- 支持按百分比缩放 -->

![logo](https://docsify.js.org/_media/icon.svg ":size=10%")
```

![logo](https://docsify.js.org/_media/icon.svg ":size=WIDTHxHEIGHT")
![logo](https://docsify.js.org/_media/icon.svg ":size=50x100")
![logo](https://docsify.js.org/_media/icon.svg ":size=100")

<!-- 支持按百分比缩放 -->

![logo](https://docsify.js.org/_media/icon.svg ":size=10%")

### 设置图片的 class 和 id

```markdown
![logo](https://docsify.js.org/_media/icon.svg ":class=someCssClass")
![logo](https://docsify.js.org/_media/icon.svg ":id=someCssId")
```

### 混写 html 和 markdown

```markdown
<details>
<summary>自我评价（点击展开）</summary>

- Abc
- Abc

</details>

<div style='color: red'>

- listitem
- listitem
- listitem

</div>
```

<details>
<summary>自我评价（点击展开）</summary>

- Abc
- Abc

</details>

<div style='color: red'>

- listitem
- listitem
- listitem

</div>
