# Studentprojekt vid LNU: Ljushetsprediktion vid kemisk blekning av pappersmassa 2022/11/03-2022/12/18

## abstract

In this project, the aim was to investigate how predictive models can be created to predict the brightness of paper pulp based on the chemical input. The project began by creating a simple model to see if the results would improve with more complex models. Afterwards, linear regression, elastic net (a combination of ridge and lasso), and random decision tree were investigated. The results showed that elastic net had the best prediction accuracy, with Random Forest as the second best.

## Introduktion

Detta är ett urdrag av koden som användes för att arbeta med detta projekt. Allt som behövs för att köra koden finns ej med. Projektet handlade om att göra en prediktions modell för pappersmassa.

## bilbotek

I `requirement` filen så finns alla bilbotek som användes vid arbetet.

## Hjälpfunktioner

I mappen `utils` finns tre stycken moduler som användes under projektets gång:

- `functions.py`: Hjälpfunktioner som var anändes för att på ett lättare sätt arbeta med datan. Innehåller t.e.x funktion för att visa och räkna ut shap värden för modelen.
- `evaluation.py`: Här finns det funktioner för att utvärdera modellen och göra beräkningar på shap värdenrna.
-`datafixing`: Funktioner som fixar till datan så att den ej innehåller t.ex felaktiga värden. Men även funktioner för att göra datan mer hanterbar.

Här förekom det också en fil `constants.py`. En fil som innehöll konstanter om datan och annat. Denna fil är ej med.